#!/usr/bin/env python3
r"""
Sequential decision making (active learning / Bayesian optimization) benchmark utilities.

This module provides a unified interface for running sequential decision making
experiments with various acquisition functions, test functions, and performance metrics.
Supports both discrete (grid) and continuous search spaces.
"""

from __future__ import annotations

import gc
import re
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from collections import defaultdict

import torch
import numpy as np


def extract_lambda_from_density_fn(density_fn_str: str) -> Optional[float]:
    """Extract lambda value from density_fn_type string.

    Examples:
        "lambda f: torch.exp(30*f)"   -> 30.0
        "lambda f: torch.exp(-2*f)"   -> -2.0
        "lambda f: torch.exp(-0.5*f)" -> -0.5
        "lambda f: torch.exp(-f)"     -> -1.0
        "lambda f: torch.exp(f)"      -> 1.0
        "lambda f: torch.exp(-f/2)"   -> -0.5
        "lambda f: torch.exp(-f/0.5)" -> -2.0
        "lambda f: torch.exp(f/2)"    -> 0.5
        "lambda f: torch.exp(-f*2)"   -> -2.0
        "lambda f: torch.exp(f*2)"    -> 2.0
    """
    # Match: torch.exp(NUMBER*f)
    match = re.search(r'torch\.exp\(\s*(-?\d+\.?\d*)\s*\*\s*f\s*\)', density_fn_str)
    if match:
        return float(match.group(1))

    # Match: torch.exp([-]f * NUMBER)
    match = re.search(r'torch\.exp\(\s*(-?)\s*f\s*\*\s*(\d+\.?\d*)\s*\)', density_fn_str)
    if match:
        sign = -1.0 if match.group(1) == '-' else 1.0
        return sign * float(match.group(2))

    # Match: torch.exp([-]f / NUMBER)
    match = re.search(r'torch\.exp\(\s*(-?)\s*f\s*/\s*(\d+\.?\d*)\s*\)', density_fn_str)
    if match:
        sign = -1.0 if match.group(1) == '-' else 1.0
        return sign / float(match.group(2))

    # Match: torch.exp(-f) or torch.exp(f)
    match = re.search(r'torch\.exp\(\s*(-?)\s*f\s*\)', density_fn_str)
    if match:
        return -1.0 if match.group(1) == '-' else 1.0

    # Raise if string looks like a Boltzmann form but didn't match any pattern
    # (avoids silent bypass of alignment check).
    if re.search(r'torch\.exp\([^)]*f[^)]*\)', density_fn_str):
        raise ValueError(
            f"Could not extract lambda from density_fn_type {density_fn_str!r}. "
            f"Supported forms: 'torch.exp(K*f)', 'torch.exp(-f*K)', 'torch.exp(-f/T)', "
            f"'torch.exp(-f)', 'torch.exp(f)'. "
            f"Either rewrite to a supported form, or set 'boltzmann_lambda' explicitly "
            f"in the SID acquisition's hyperparameters."
        )

    return None


def density_fn_str_for_lambda(lambda_val: float) -> str:
    """Build a density_fn_type string with the given effective lambda.

    Output uses the K*f format so extract_lambda_from_density_fn parses it back correctly.
    """
    return f"lambda f: torch.exp({lambda_val}*f)"


def apply_lambda_val_to_acq(acq_name: str, acq_hyperparams: dict, lambda_val: float) -> None:
    """Mutate acq_hyperparams to align with lambda_val (SID acqs and HAL only).

    For SID-iVAR / SID-EPIG: sets boltzmann_lambda = lambda_val.
    For HAL: sets temperature = 1/|lambda_val| and boltzmann_sign = sign(lambda_val).
    Other acqs: no-op.
    """
    name_lower = acq_name.lower()
    if "sid-ivar" in name_lower or "sid-epig" in name_lower:
        acq_hyperparams["boltzmann_lambda"] = lambda_val
    elif name_lower == "hal":
        acq_hyperparams["temperature"] = 1.0 / abs(lambda_val)
        acq_hyperparams["boltzmann_sign"] = 1.0 if lambda_val > 0 else -1.0


def check_lambda_alignment(acq_name: str, acq_hyperparams: dict, density_fn_str: str) -> Optional[float]:
    """Verify acq's effective lambda matches the eval density_fn's lambda.

    Applies to SID acquisitions (iVAR + EPIG) and HAL — all use a Boltzmann-form
    internal target whose sharpness must match the eval target.

    Args:
        acq_name: acquisition name (e.g., "AB-SID-iVAR", "HAL", "mu-SID-EPIG").
        acq_hyperparams: acquisition hyperparameter dict.
        density_fn_str: density_fn_type string from config.

    Returns:
        The effective acq lambda if check applies, else None (acq has no lambda).

    Raises:
        ValueError: if magnitudes or signs mismatch.
    """
    name_lower = acq_name.lower()
    is_sid = "sid-ivar" in name_lower or "sid-epig" in name_lower
    is_hal = name_lower == "hal"
    if not (is_sid or is_hal):
        return None

    extracted_lambda = extract_lambda_from_density_fn(density_fn_str)
    acq_lambda = acq_hyperparams.get("boltzmann_lambda", None)
    if acq_lambda is None:
        # Class default: temperature=1.0, sign=-1.0 for SID acqs and HAL.
        t = acq_hyperparams.get("temperature", 1.0)
        s = acq_hyperparams.get("boltzmann_sign", -1.0)
        acq_lambda = s / t

    if extracted_lambda is None:
        return acq_lambda  # density_fn doesn't expose a parseable lambda; skip check.

    if abs(abs(acq_lambda) - abs(extracted_lambda)) > 1e-6:
        raise ValueError(
            f"Mismatch ({acq_name}): density_fn_type has lambda={extracted_lambda}, "
            f"but effective acq lambda={acq_lambda}. "
            f"Either align them in the config, or pass --lambda_val {extracted_lambda} "
            f"to override at runtime."
        )
    if (acq_lambda > 0) != (extracted_lambda > 0):
        raise ValueError(
            f"Sign mismatch ({acq_name}): density_fn_type has lambda={extracted_lambda} "
            f"but effective acq lambda={acq_lambda}."
        )
    return acq_lambda


from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import CandidateGenerationError
from botorch.optim import optimize_acqf
from botorch.test_functions.base import BaseTestProblem
from .model_builder import build_and_fit_gp


def generate_grid_candidates(bounds: Tensor, n_grid_per_dim: int) -> Tensor:
    """Generate grid candidates for discrete search space.

    Args:
        bounds: (2, d) lower and upper bounds
        n_grid_per_dim: Number of grid points per dimension

    Returns:
        (n_grid_per_dim^d, d) tensor of candidates
    """
    dim = bounds.shape[1]
    grids_1d = [
        torch.linspace(bounds[0, i], bounds[1, i], n_grid_per_dim,
                      dtype=bounds.dtype, device=bounds.device)
        for i in range(dim)
    ]
    meshgrids = torch.meshgrid(*grids_1d, indexing='ij')
    candidates = torch.stack([g.flatten() for g in meshgrids], dim=-1)
    return candidates


def _get_variance_constraint(model, acq_function) -> Optional[Tuple]:
    """Get variance constraint for continuous optimization if applicable.

    Returns:
        Tuple of (constraint_fn, is_intra_point) for optimize_acqf, or None.
    """
    if not hasattr(acq_function, '_constraint_type') or acq_function._constraint_type is None:
        return None

    # Access reference_points OUTSIDE no_grad context (may trigger SMC sampling of the reference points)
    ref_pts = acq_function.reference_points

    # Compute threshold from reference points
    with torch.no_grad():
        ref_vars = model.posterior(ref_pts).variance.squeeze(-1)

    if acq_function._constraint_type == "energy_weighted_variance":
        # For continuous acquisition functions, reference points are SMC samples
        # from the target distribution P. The correct estimator for E_P[σ²] is
        # simply mean(σ²) (direct Monte Carlo with equal weights).
        # For discrete acquisition functions, reference points are a grid and we
        # need importance weighting: E_P[σ²] = Σ P(x_i) · σ²(x_i).
        # Using weighted average for SMC samples is DOUBLE-WEIGHTING.
        is_continuous = hasattr(acq_function, 'space') and not acq_function.space.is_discrete
        if is_continuous:
            # SMC samples from P: use equal weights (direct Monte Carlo)
            threshold = ref_vars.mean()
        else:
            # Grid points: use importance weighting
            weights = acq_function.target_distribution
            threshold = (weights * ref_vars).sum()
    else:
        return None

    def variance_constraint(X: Tensor) -> Tensor:
        """Constraint: σ²(x) - threshold ≥ 0 (feasible = high variance).

        Args:
            X: (d,) single point (is_intra_point=True)

        Returns:
            scalar constraint value where ≥ 0 means feasible
        """
        # is_intra_point=True means X is always (d,)
        posterior = model.posterior(X.unsqueeze(0))  # (1, d) -> posterior
        var = posterior.variance.squeeze()  # scalar
        return var - threshold

    return (variance_constraint, True)


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    candidates: Tensor,
    verbose: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Optimize acquisition function over discrete candidates using argmax.

    Uses 2D input (N, d) instead of 3D (N, 1, d) to avoid a memory leak in
    GPyTorch/linear_operator's batch dimension handling when creating new models.

    Args:
        acq_function: Acquisition function to optimize
        candidates: (n_candidates, d) tensor of candidates
        verbose: If True, log memory usage at key points

    Returns:
        Tuple of (best_candidate, acq_value) where best_candidate is (1, d)
    """
    from sidal import PosteriorVariance

    with torch.no_grad():
        # Use 2D input (N, d) to avoid GPyTorch memory leak with 3D (N, 1, d)
        if isinstance(acq_function, PosteriorVariance):
            posterior = acq_function.model.posterior(candidates)
            acq_values = posterior.variance.squeeze(-1)
            if not acq_function.maximize:
                acq_values = -acq_values
        else:
            # Fallback for other acquisition functions
            X = candidates.unsqueeze(1)  # (N, 1, d)
            acq_values = acq_function(X)

    # Find best (handle -inf values from constraint sets)
    valid_mask = torch.isfinite(acq_values)
    if not valid_mask.any():
        idx = torch.randint(len(candidates), (1,))
        return candidates[idx].unsqueeze(0), torch.tensor([float('-inf')])

    acq_values_masked = acq_values.clone()
    acq_values_masked[~valid_mask] = float('-inf')
    idx = acq_values_masked.argmax()

    return candidates[idx].unsqueeze(0), acq_values[idx].unsqueeze(0)


def get_acquisition_function(
    acq_name: str,
    model: Any,
    candidates: Optional[Tensor] = None,
    bounds: Optional[Tensor] = None,
    iteration: int = 0,
    acq_hyperparameters: Optional[Dict[str, Any]] = None,
) -> AcquisitionFunction:
    """Get acquisition function by name.

    Args:
        acq_name: Name of acquisition function:
            - "RS": Random Sampling
            - "US": Uncertainty Sampling (posterior variance)
            - "IMSE": Integrated Mean Squared Error
            - "EPIG": Expected Predictive Information Gain
            - "HAL": Generic heuristic active learning baseline (GHAL in the paper)
            - "AB-SID-iVAR": Analytic Boltzmann SID with integrated variance (ours)
            - "TS-SID-iVAR": Thompson-sampling SID with integrated variance (ours)
            - "mu-SID-iVAR", "mu-SID-EPIG": plug-in posterior-mean ablations
            For SID methods the constraint set is selected via
            acq_hyperparameters["constraint_type"] ("energy_weighted_variance" or None).
        model: Fitted GP model
        candidates: Discrete candidates (for discrete domain). If None, uses continuous.
        bounds: (2, d) bounds tensor (required for continuous domain)
        iteration: Current iteration number (used for SMC seed in continuous domain)
        acq_hyperparameters: Optional hyperparameters dict. For continuous SID-iVAR,
            requires: "n_smc_particles", "n_mutation_steps", "smc_seed"

    Returns:
        Acquisition function instance
    """
    if acq_hyperparameters is None:
        acq_hyperparameters = {}

    acq_name_lower = acq_name.lower()
    is_discrete = candidates is not None

    # Random Sampling
    if acq_name_lower == "rs":
        class RandomSampling(AcquisitionFunction):
            _is_random_sampling = True

            def forward(self, X):
                return torch.rand(X.shape[:-2], device=X.device, dtype=X.dtype)

        return RandomSampling(model=model)

    # Uncertainty Sampling (posterior variance)
    elif acq_name_lower == "us":
        from sidal import PosteriorVariance
        return PosteriorVariance(model=model, maximize=True)

    # Integrated Mean Squared Error (using optimized version with memory fixes)
    elif acq_name_lower == "imse":
        from sidal import qNegIntegratedPosteriorVariance, DiscreteDesignSpace, ContinuousDesignSpace
        if is_discrete:
            space = DiscreteDesignSpace(candidates=candidates)
        else:
            if bounds is None:
                raise ValueError("IMSE continuous requires bounds")
            # For IMSE continuous, use uniform reference points (re-sampled each iteration)
            required_keys = ["n_ref_points_continuous", "ref_seed_continuous"]
            for key in required_keys:
                if key not in acq_hyperparameters:
                    raise ValueError(f"Continuous IMSE requires '{key}' in acq_hyperparameters")

            from sidal import uniform_reference_point_generator
            ref_seed = acq_hyperparameters["ref_seed_continuous"] + iteration  # Different per iteration
            space = ContinuousDesignSpace(
                bounds=bounds,
                reference_point_generator=uniform_reference_point_generator,
                reference_point_generator_kwargs={
                    "n_samples": acq_hyperparameters["n_ref_points_continuous"],
                    "seed": ref_seed,
                },
            )
        return qNegIntegratedPosteriorVariance(model=model, space=space)

    # Expected Predictive Information Gain
    elif acq_name_lower == "epig":
        from sidal import qExpectedPredictiveInformationGain, DiscreteDesignSpace, ContinuousDesignSpace
        if is_discrete:
            space = DiscreteDesignSpace(candidates=candidates)
        else:
            if bounds is None:
                raise ValueError("EPIG continuous requires bounds")
            # For EPIG continuous, use uniform reference points (re-sampled each iteration)
            required_keys = ["n_ref_points_continuous", "ref_seed_continuous"]
            for key in required_keys:
                if key not in acq_hyperparameters:
                    raise ValueError(f"Continuous EPIG requires '{key}' in acq_hyperparameters")

            from sidal import uniform_reference_point_generator
            ref_seed = acq_hyperparameters["ref_seed_continuous"] + iteration  # Different per iteration
            space = ContinuousDesignSpace(
                bounds=bounds,
                reference_point_generator=uniform_reference_point_generator,
                reference_point_generator_kwargs={
                    "n_samples": acq_hyperparameters["n_ref_points_continuous"],
                    "seed": ref_seed,
                },
            )
        return qExpectedPredictiveInformationGain(model=model, space=space)

    # SID-iVAR acquisition functions
    # Naming:
    #   - AB-SID-iVAR: Analytic Boltzmann (zeroth_order_analytic: exp(λμ + λ²σ²/2))
    #   - TS-SID-iVAR: Thompson Sampling
    #   - mu-SID-iVAR: Mean-only (exp(λμ)) - ablation study
    #   - mu-SID-EPIG: Mean-only EPIG variant - ablation study
    # The constraint set is given by acq_hyperparameters["constraint_type"].
    elif acq_name_lower in ["ab-sid-ivar", "ts-sid-ivar", "mu-sid-ivar", "mu-sid-epig"]:
        from sidal import (
            AB_SID_iVAR_Discrete,
            AB_SID_iVAR_Continuous,
            TS_SID_iVAR_Discrete,
            TS_SID_iVAR_Continuous,
            Mu_SID_iVAR_Continuous,
            Mu_SID_EPIG_Continuous,
            Mu_SID_EPIG_Discrete,
            DiscreteDesignSpace,
            ContinuousDesignSpace,
        )

        # Determine variant type
        use_thompson = acq_name_lower.startswith("ts-sid-ivar")
        use_mean_only = acq_name_lower.startswith("mu-sid-ivar")
        use_mu_epig = acq_name_lower.startswith("mu-sid-epig")

        if is_discrete:
            space = DiscreteDesignSpace(candidates=candidates)

            # Convert boltzmann_lambda to temperature + boltzmann_sign if provided
            # Paper uses λ directly: P(x) ∝ exp(λf(x))
            # Code uses: lambda_over_T = boltzmann_sign / temperature
            # So: temperature = 1/|λ|, boltzmann_sign = sign(λ)
            if "boltzmann_lambda" in acq_hyperparameters:
                bl = acq_hyperparameters["boltzmann_lambda"]
                acq_hyperparameters["temperature"] = 1.0 / abs(bl)
                acq_hyperparameters["boltzmann_sign"] = 1.0 if bl > 0 else -1.0

            # Build kwargs only for provided hyperparameters
            sid_ivar_keys = ["temperature", "boltzmann_sign", "constraint_type", "stabilize_log_density", "num_rff_features", "sample_approach"]
            sid_ivar_kwargs = {k: acq_hyperparameters[k] for k in sid_ivar_keys if k in acq_hyperparameters}

            if use_thompson:
                return TS_SID_iVAR_Discrete(space=space, model=model, **sid_ivar_kwargs)
            elif use_mu_epig:
                # mu-SID-EPIG_Discrete doesn't accept sample_approach kwarg
                mu_epig_kwargs = {k: v for k, v in sid_ivar_kwargs.items() if k != "sample_approach"}
                return Mu_SID_EPIG_Discrete(space=space, model=model, **mu_epig_kwargs)
            else:
                # mu-SID-iVAR discrete not implemented, use AB for now
                return AB_SID_iVAR_Discrete(space=space, model=model, **sid_ivar_kwargs)
        else:
            # Continuous domain - requires SMC parameters
            if bounds is None:
                raise ValueError("SID-iVAR continuous requires bounds")

            required_smc_keys = ["n_smc_particles", "n_mutation_steps", "smc_seed"]
            for key in required_smc_keys:
                if key not in acq_hyperparameters:
                    raise ValueError(
                        f"Continuous SID-iVAR requires '{key}' in acq_hyperparameters"
                    )

            from sidal import smc_reference_point_generator

            # Compute iteration-dependent seed for reproducibility without bias
            smc_seed = acq_hyperparameters["smc_seed"] + iteration

            # SMC temperature schedule: fixed (n_temperatures) or adaptive (ESS-based)
            # Default: fixed schedule with n_temperatures steps
            n_temperatures = acq_hyperparameters.get("n_temperatures", 20)
            adaptive_temperatures = acq_hyperparameters.get("adaptive_temperatures", False)
            mutation_kernel = acq_hyperparameters.get("mutation_kernel", "rwm")

            space = ContinuousDesignSpace(
                bounds=bounds,
                reference_point_generator=smc_reference_point_generator,
                reference_point_generator_kwargs={
                    "n_samples": acq_hyperparameters["n_smc_particles"],
                    "n_mutation_steps": acq_hyperparameters["n_mutation_steps"],
                    "n_temperatures": n_temperatures,
                    "adaptive_temperatures": adaptive_temperatures,
                    "seed": smc_seed,
                    "mutation_kernel": mutation_kernel,
                    "verbose": True,  # Print SMC progress to logs
                },
            )

            # Convert boltzmann_lambda to temperature + boltzmann_sign if provided
            if "boltzmann_lambda" in acq_hyperparameters:
                bl = acq_hyperparameters["boltzmann_lambda"]
                acq_hyperparameters["temperature"] = 1.0 / abs(bl)
                acq_hyperparameters["boltzmann_sign"] = 1.0 if bl > 0 else -1.0

            # Build kwargs only for provided hyperparameters
            sid_ivar_keys = ["temperature", "boltzmann_sign", "constraint_type", "stabilize_log_density", "num_rff_features", "sample_approach"]
            sid_ivar_kwargs = {k: acq_hyperparameters[k] for k in sid_ivar_keys if k in acq_hyperparameters}

            if use_thompson:
                return TS_SID_iVAR_Continuous(space=space, model=model, **sid_ivar_kwargs)
            elif use_mu_epig:
                return Mu_SID_EPIG_Continuous(space=space, model=model, **sid_ivar_kwargs)
            elif use_mean_only:
                return Mu_SID_iVAR_Continuous(space=space, model=model, **sid_ivar_kwargs)
            else:
                return AB_SID_iVAR_Continuous(space=space, model=model, **sid_ivar_kwargs)

    else:
        raise ValueError(
            f"Unknown acquisition function: {acq_name}. "
            f"Supported: 'RS', 'US', 'IMSE', 'EPIG', "
            f"'AB-SID-iVAR', 'TS-SID-iVAR', 'mu-SID-iVAR', 'mu-SID-EPIG'. "
            f"For HAL-style experiments, use run_hal_experiment() instead."
        )


def run_experiment(
    test_function: Union[BaseTestProblem, Callable[[Tensor], Tensor]],
    bounds: Tensor,
    acq_name: str,
    seed: int,
    n_init: int,
    n_iterations: int,
    density_fn: Callable[[Tensor], Tensor],
    model_config: Dict[str, Any],
    design_space_type: str = "discrete",
    n_grid_per_dim: int = 20,
    acq_hyperparameters: Optional[Dict[str, Any]] = None,
    init_strategy: str = "random",
    performance_metrics: Optional[List[str]] = None,
    evaluation_grid: Optional[Tensor] = None,
    n_eval_points: int = 1000,
    eval_seed: int = 42,
    save_reference_points: bool = False,
    verbose: bool = True,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    **kwargs,
) -> Dict[str, Any]:
    r"""Run a single benchmarking experiment.

    This function provides a unified interface for benchmarking different acquisition
    functions on test problems. Supports both discrete and continuous search spaces.

    Args:
        test_function: Either a BoTorch BaseTestProblem or a callable that maps
            Tensor of shape `(n, d)` to Tensor of shape `(n,)` or `(n, 1)`.
        bounds: A `(2, d)` tensor of lower and upper bounds for each dimension.
        acq_name: Name of acquisition function:
            - "RS": Random Sampling
            - "US": Uncertainty Sampling
            - "IMSE": Integrated Mean Squared Error
            - "EPIG": Expected Predictive Information Gain
            - "HAL": Generic heuristic active learning baseline (GHAL in the paper)
            - "AB-SID-iVAR": Analytic Boltzmann SID with integrated variance (ours)
            - "TS-SID-iVAR": Thompson-sampling SID with integrated variance (ours)
            - "mu-SID-iVAR", "mu-SID-EPIG": plug-in posterior-mean ablations
        seed: Random seed for reproducibility. Controls:
            - Initial design of experiments (DOE) point selection
            - Random noise added to observations
            - Random candidate selection when acquisition returns all -inf
            Note: Different seeds should be used for different repetitions/trials.
        n_init: Number of initial random samples.
        n_iterations: Number of sequential acquisition iterations.
        density_fn: Density function for weighted metrics (e.g., `lambda f: torch.exp(-f)`).
            Maps function values to density weights. Required for proper benchmarking
            of active learning methods that target specific regions.
        model_config: Dictionary of GP model configuration. Required keys:
            - "kernel": str - Kernel type ("rbf", "matern52", "matern32", "matern12")
            - "use_input_transform": bool - Apply Normalize transform to inputs
            - "use_outcome_transform": bool - Apply Standardize transform to outputs
            - "use_default_kernel_priors": bool - Use BoTorch's default MAP priors
            - "noise_level": float - Observation noise variance
            - "fix_noise": bool - If True, freeze the noise parameter
            Optional keys:
            - "fixed_lengthscale": List[float] - Fixed lengthscale values (ARD)
            - "fixed_outputscale": float - Fixed outputscale value
        design_space_type: Type of design space. Options:
            - "discrete": Grid-based search with argmax over candidates
            - "continuous": Continuous optimization via optimize_acqf
            - "pool": Pool-based AL with pre-featurized candidates (requires kwargs)
        n_grid_per_dim: Grid points per dimension (only used if design_space_type="discrete").
        acq_hyperparameters: Optional hyperparameters for acquisition function.
        init_strategy: Initial sampling strategy ("random", "sobol", "latin_hypercube").
        performance_metrics: List of metrics to compute. Options:
            - "mse": Mean squared error on evaluation grid
            - "weighted_mse": Weighted MSE with density_fn
            - "nlml": Negative log predictive likelihood
            - "weighted_nlml": Weighted NLML with density_fn
            - "mae": Mean absolute error
            - "max_error": Maximum absolute error
            If None, defaults to ["mse"].
        evaluation_grid: Optional tensor of shape `(n_eval, d)` for computing metrics.
        n_eval_points: Number of evaluation points if evaluation_grid is None
            (only used for continuous domains).
        eval_seed: Random seed for sampling evaluation grid (only used for continuous
            domains when evaluation_grid is None). Ensures reproducible evaluation
            across different runs. Default: 42.
        verbose: If True, print progress information.
        num_restarts: Number of restarts for continuous optimization (optimize_acqf).
            Default: 20. Only used when design_space_type="continuous".
        raw_samples: Number of raw samples for initializing continuous optimization.
            Default: 1024. Only used when design_space_type="continuous".

    Returns:
        A dictionary containing the experiment history:
            - "X": List of training inputs at each iteration
            - "Y": List of training targets at each iteration
            - "candidates_selected": List of selected points
            - "acq_values": List of acquisition values
            - "metrics": Dict mapping metric names to lists of values
            - "model_hyperparameters": List of model hyperparameters
            - "config": Dict with experiment configuration
    """
    # Validate model_config
    required_model_keys = ["kernel", "use_input_transform", "use_outcome_transform",
                          "use_default_kernel_priors", "noise_level", "fix_noise"]
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"model_config must contain '{key}'")

    # Extract model config
    kernel = model_config["kernel"]
    use_input_transform = model_config["use_input_transform"]
    use_outcome_transform = model_config["use_outcome_transform"]
    use_default_kernel_priors = model_config["use_default_kernel_priors"]
    noise_level = model_config["noise_level"]
    fix_noise = model_config["fix_noise"]
    fixed_lengthscale = model_config.get("fixed_lengthscale", None)
    fixed_outputscale = model_config.get("fixed_outputscale", None)
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = bounds.shape[1]
    dtype = bounds.dtype
    device = bounds.device

    # Prepare evaluation function
    if test_function is None:
        # Pool-based: no test function, use pre-computed Y values from kwargs
        eval_fn = None
    elif isinstance(test_function, BaseTestProblem):
        def eval_fn(X: Tensor) -> Tensor:
            y = test_function(X)
            return y.unsqueeze(-1) if y.dim() == 1 else y
    else:
        def eval_fn(X: Tensor) -> Tensor:
            y = test_function(X)
            return y.unsqueeze(-1) if y.dim() == 1 else y

    # Generate candidates for discrete/pool search
    candidates = None
    if design_space_type == "discrete":
        candidates = generate_grid_candidates(bounds, n_grid_per_dim)
        if verbose:
            print(f"Discrete search space: {len(candidates)} candidates")
    elif design_space_type == "pool":
        # Pool-based: get candidates from kwargs
        candidates = kwargs.get("pool_X")
        if candidates is None:
            raise ValueError("design_space_type='pool' requires 'pool_X' in kwargs")
        if verbose:
            print(f"Pool-based search space: {len(candidates)} candidates")

    # Generate evaluation grid for metrics
    if performance_metrics is None:
        performance_metrics = ["mse"]

    if evaluation_grid is None and len(performance_metrics) > 0:
        if design_space_type == "discrete":
            # In discrete setting, evaluate on all grid candidates
            evaluation_grid = candidates
            if verbose:
                print(f"Evaluation grid: {len(evaluation_grid)} points (all candidates)")
        elif design_space_type == "pool":
            # In pool setting, use held-out test set from kwargs
            evaluation_grid = kwargs.get("test_X")
            eval_Y_true = kwargs.get("test_Y")
            if evaluation_grid is None:
                raise ValueError("design_space_type='pool' requires 'test_X' in kwargs for metrics")
            if verbose:
                print(f"Evaluation grid: {len(evaluation_grid)} held-out test points")
        else:
            # In continuous setting, use random evaluation points
            torch.manual_seed(eval_seed)  # Fixed seed for reproducible evaluation grid
            evaluation_grid = torch.rand(n_eval_points, dim, dtype=dtype, device=device)
            evaluation_grid = bounds[0] + evaluation_grid * (bounds[1] - bounds[0])
            if verbose:
                print(f"Evaluation grid: {len(evaluation_grid)} random points (eval_seed={eval_seed})")

    # Precompute true function values on evaluation grid
    eval_Y_true = None
    if evaluation_grid is not None:
        if kwargs.get("precomputed_eval_Y") is not None:
            eval_Y_true = kwargs["precomputed_eval_Y"].squeeze()
            if verbose:
                print(f"Using precomputed eval_Y_true ({len(eval_Y_true)} points)")
        elif design_space_type == "pool":
            # Pool-based: use pre-computed test_Y from kwargs
            eval_Y_true = kwargs.get("test_Y")
            if eval_Y_true is None:
                raise ValueError("design_space_type='pool' requires 'test_Y' in kwargs for metrics")
            eval_Y_true = eval_Y_true.squeeze()
        else:
            # Ground-truth evaluation must not consume the experiment's random
            # stream (e.g. MACE draws from torch's global RNG); otherwise the
            # observation noise of every later query depends on it.
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                eval_Y_true = eval_fn(evaluation_grid).squeeze()

    # Generate initial data
    if design_space_type == "pool":
        # Pool-based: sample from pool and lookup Y values
        pool_Y = kwargs.get("pool_Y")
        if pool_Y is None:
            raise ValueError("design_space_type='pool' requires 'pool_Y' in kwargs")
        torch.manual_seed(seed)
        init_indices = torch.randperm(len(candidates))[:n_init]
        train_X = candidates[init_indices]
        train_Y = pool_Y[init_indices].unsqueeze(-1) if pool_Y.dim() == 1 else pool_Y[init_indices]
        # Track available pool indices
        pool_mask = torch.ones(len(candidates), dtype=torch.bool)
        pool_mask[init_indices] = False
    else:
        train_X = _generate_initial_data(bounds, n_init, init_strategy, seed, candidates if design_space_type == "discrete" else None)
        train_Y = eval_fn(train_X)
        pool_mask = None
        pool_Y = None

    # Add noise to observations
    if noise_level > 0:
        train_Y = train_Y + torch.randn_like(train_Y) * (noise_level ** 0.5)

    # Initialize history
    # For pool-based: skip per-iteration X/Y history to save space
    history = {
        "X": [] if design_space_type == "pool" else [train_X.clone()],
        "Y": [] if design_space_type == "pool" else [train_Y.clone()],
        "candidates_selected": [],  # Newly selected points per iteration
        "acq_values": [],
        "reference_points": [],  # Store reference points per iteration (for replay in notebooks)
        "model_hyperparameters": [],
        "metrics": defaultdict(list),
        "timing": {  # Timing information per iteration
            "iteration_total": [],      # Total iteration time
            "gp_fitting": [],           # GP fitting time
            "acq_construction": [],     # Acquisition function construction
            "acq_optimization": [],     # Acquisition function optimization
            "metric_computation": [],   # Metric computation time
            "smc_sampling": [],         # SMC sampling time (for continuous SID-iVAR)
        },
        "smc_info": [],  # Detailed SMC info per iteration (particles, ESS, kernel, etc.)
        "config": {
            "acq_name": acq_name,
            "seed": seed,
            "n_init": n_init,
            "n_iterations": n_iterations,
            "design_space_type": design_space_type,
            "n_grid_per_dim": n_grid_per_dim,
            "dim": dim,
            "bounds": bounds.tolist() if bounds is not None else None,
        },
        # Store model configuration separately for clarity
        "model_config": model_config,
        # Store acquisition hyperparameters (e.g., constraint_type, base_approximation, etc.)
        "acq_hyperparameters": acq_hyperparameters if acq_hyperparameters else {},
    }

    # For pool-based: store initial training data (since X/Y history is skipped)
    if design_space_type == "pool":
        history["init_X"] = train_X.clone()
        history["init_Y"] = train_Y.clone()

    # Build model hyperparameters dict
    model_hyperparams = {
        "noise": noise_level,
        "freeze_noise": fix_noise,
    }
    if fixed_lengthscale is not None:
        model_hyperparams["lengthscale"] = fixed_lengthscale
        model_hyperparams["freeze_lengthscale"] = True
    if fixed_outputscale is not None:
        model_hyperparams["outputscale"] = fixed_outputscale
        model_hyperparams["freeze_outputscale"] = True

    if verbose:
        print(f"\nStarting experiment: {acq_name}")
        print(f"  Seed: {seed}, Noise: {noise_level}, Fixed: {fix_noise}")
        print(f"  Design space: {design_space_type}, Init: {n_init}, Iterations: {n_iterations}")
        print("=" * 60)

    # Import get_model_hyperparameters for storing hyperparameters
    from .model_builder import get_model_hyperparameters

    # Compute initial metrics (iteration 0 = before any acquisition)
    if evaluation_grid is not None and len(performance_metrics) > 0:
        initial_model = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=bounds,
            kernel=kernel,
            hyperparameters=model_hyperparams,
            use_input_transform=use_input_transform,
            use_outcome_transform=use_outcome_transform,
            use_default_kernel_priors=use_default_kernel_priors,
            verbose=False,
        )
        history["model_hyperparameters"].append(get_model_hyperparameters(initial_model))

        metrics_dict = _compute_metrics(
            model=initial_model,
            evaluation_grid=evaluation_grid,
            eval_Y_true=eval_Y_true,
            metric_names=performance_metrics,
            density_fn=density_fn,
        )
        for metric_name, metric_value in metrics_dict.items():
            history["metrics"][metric_name].append(metric_value)

        if verbose:
            metric_strs = [f"{k}={v:.6f}" for k, v in metrics_dict.items()]
            print(f"Initial metrics (iter 0): {', '.join(metric_strs)}")

    # Main loop
    import time as time_module
    model_builder_fn = kwargs.get("model_builder_fn")
    for iteration in range(n_iterations):
        iter_start_time = time_module.perf_counter()
        if verbose:
            print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Build and fit GP
        gp_start_time = time_module.perf_counter()
        if model_builder_fn is not None:
            # Use custom model builder (e.g., for Tanimoto kernel in pool-based AL)
            model = model_builder_fn(train_X, train_Y)
        else:
            model = build_and_fit_gp(
                train_X=train_X,
                train_Y=train_Y,
                input_bounds=bounds,
                kernel=kernel,
                hyperparameters=model_hyperparams,
                use_input_transform=use_input_transform,
                use_outcome_transform=use_outcome_transform,
                use_default_kernel_priors=use_default_kernel_priors,
                verbose=False,
            )
        gp_time = time_module.perf_counter() - gp_start_time

        # Get acquisition function (includes SMC sampling for continuous AB-SID-iVAR)
        acq_construct_start = time_module.perf_counter()
        acq_function = get_acquisition_function(
            acq_name=acq_name,
            model=model,
            candidates=candidates,
            bounds=bounds,
            iteration=iteration,
            acq_hyperparameters=acq_hyperparameters,
        )
        acq_construct_time = time_module.perf_counter() - acq_construct_start

        # Optimize acquisition
        acq_opt_start = time_module.perf_counter()
        if hasattr(acq_function, '_is_random_sampling'):
            # Random sampling
            if design_space_type in ("discrete", "pool"):
                # For pool, only sample from available candidates
                if design_space_type == "pool" and pool_mask is not None:
                    available_idx = pool_mask.nonzero().squeeze(-1)
                    rand_idx = torch.randint(len(available_idx), (1,))
                    idx = available_idx[rand_idx]
                else:
                    idx = torch.randint(len(candidates), (1,))
                new_x = candidates[idx]
            else:
                new_x = torch.rand(1, dim, dtype=dtype, device=device)
                new_x = bounds[0] + new_x * (bounds[1] - bounds[0])
            acq_value = torch.tensor([0.0])
        elif design_space_type in ("discrete", "pool"):
            # For pool, only evaluate on available candidates
            if design_space_type == "pool" and pool_mask is not None:
                available_candidates = candidates[pool_mask]
                new_x, acq_value = optimize_acqf_discrete(acq_function, available_candidates)
            else:
                new_x, acq_value = optimize_acqf_discrete(acq_function, candidates)
        else:
            # Continuous optimization
            optimize_kwargs = {
                "acq_function": acq_function,
                "bounds": bounds,
                "q": 1,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
            }

            # Get variance constraint if applicable
            constraint_result = _get_variance_constraint(model, acq_function)
            constraint_fn = None
            nl_constraints = []
            if constraint_result is not None:
                constraint_fn, is_intra_point = constraint_result
                nl_constraints.append((constraint_fn, is_intra_point))

            if nl_constraints:
                optimize_kwargs["nonlinear_inequality_constraints"] = nl_constraints

                # Generate feasible initial conditions via rejection sampling
                def _gen_feasible_ics(acq_function, bounds, num_restarts, raw_samples, **kwargs):
                    """Generate feasible initial conditions for constrained optimization."""
                    q = 1
                    d = bounds.shape[1]
                    feasible = []
                    max_attempts = raw_samples * 10
                    for _ in range(max_attempts):
                        x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(d)
                        if all(c(x) >= 0 for c, _ in nl_constraints):
                            feasible.append(x)
                        if len(feasible) >= raw_samples:
                            break
                    if len(feasible) < num_restarts:
                        raise RuntimeError(f"Could not generate enough feasible ICs: {len(feasible)}/{num_restarts}")
                    feasible = torch.stack(feasible)  # (n_feasible, d)
                    # Evaluate acq and pick top num_restarts
                    with torch.no_grad():
                        acq_vals = acq_function(feasible.unsqueeze(1))  # (n_feasible,)
                    top_idx = torch.argsort(acq_vals, descending=True)[:num_restarts]
                    return feasible[top_idx].unsqueeze(1)  # (num_restarts, 1, d)

                optimize_kwargs["ic_generator"] = _gen_feasible_ics

            # Use reference points as initialization pool if available
            # (for SID-iVAR on continuous spaces, reference points are SMC samples)
            if hasattr(acq_function, 'reference_points'):
                ref_pts = acq_function.reference_points  # (n_ref, d)
                ref_pts_q = ref_pts.unsqueeze(1)  # (n_ref, 1, d)

                # Filter to points satisfying all constraints (variance + distance)
                if nl_constraints:
                    with torch.no_grad():
                        feasible_mask = torch.ones(len(ref_pts), dtype=torch.bool)
                        for c_fn, _ in nl_constraints:
                            c_vals = torch.stack([c_fn(pt) for pt in ref_pts])
                            feasible_mask &= (c_vals >= 0)
                    ref_pts_q = ref_pts_q[feasible_mask]

                # Select top num_restarts by acquisition value
                if len(ref_pts_q) > 0:
                    with torch.no_grad():
                        acq_vals = acq_function(ref_pts_q)
                    n_select = min(num_restarts, len(ref_pts_q))
                    top_indices = torch.argsort(acq_vals, descending=True)[:n_select]
                    top_init = ref_pts_q[top_indices]
                    optimize_kwargs["batch_initial_conditions"] = top_init
                    optimize_kwargs["raw_samples"] = 0
                    # Use only the feasible reference points as restarts. The feasible
                    # set is never empty (tau is the mean of sigma^2 over the reference
                    # points), but it can hold fewer than num_restarts points; without
                    # this, BoTorch tops up with the ic_generator, which gets
                    # raw_samples=0 and so cannot find any point.
                    optimize_kwargs["num_restarts"] = n_select

            try:
                new_x, acq_value = optimize_acqf(**optimize_kwargs)
            except CandidateGenerationError:
                # SLSQP moved every starting point outside the variance constraint.
                # The starting points are feasible by construction, so fall back to
                # the best of them rather than aborting the run.
                ics = optimize_kwargs.get("batch_initial_conditions")
                if ics is None:
                    raise
                with torch.no_grad():
                    ic_vals = acq_function(ics)
                best = ic_vals.argmax()
                new_x, acq_value = ics[best], ic_vals[best]
                print("  Warning: SLSQP returned no feasible candidate; "
                      "using the best feasible starting point")
        acq_opt_time = time_module.perf_counter() - acq_opt_start

        # Ensure new_x has shape (1, d) - squeeze any extra dimensions
        if new_x.dim() > 2:
            new_x = new_x.squeeze()
        if new_x.dim() == 1:
            new_x = new_x.unsqueeze(0)

        # Evaluate test function or lookup from pool
        if design_space_type == "pool":
            # Find index in original candidates and lookup Y
            match_mask = (candidates == new_x).all(dim=1)
            selected_idx = match_mask.nonzero().squeeze(-1).item()
            new_y = pool_Y[selected_idx:selected_idx+1].unsqueeze(-1) if pool_Y.dim() == 1 else pool_Y[selected_idx:selected_idx+1]
            # Mark as used in pool
            pool_mask[selected_idx] = False
        else:
            new_y = eval_fn(new_x)

        # Add noise (not for pool - labels are fixed)
        if noise_level > 0 and design_space_type != "pool":
            new_y = new_y + torch.randn_like(new_y) * (noise_level ** 0.5)

        if verbose:
            x_str = ", ".join([f"{new_x[0, i].item():.4f}" for i in range(min(dim, 5))])
            if dim > 5:
                x_str += ", ..."
            print(f"  Selected: [{x_str}], Value: {new_y.item():.4f}")

        # Update training data
        train_X = torch.cat([train_X, new_x], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)

        # Record selected point
        history["candidates_selected"].append(new_x.clone())
        history["acq_values"].append(acq_value.item())

        # Store reference points if enabled (for notebook replay)
        if save_reference_points and hasattr(acq_function, 'reference_points'):
            history["reference_points"].append(acq_function.reference_points.clone())

        # Capture SMC timing info (for continuous SID-iVAR)
        if hasattr(acq_function, '_last_smc_info') and acq_function._last_smc_info is not None:
            smc_info = acq_function._last_smc_info
            history["timing"]["smc_sampling"].append(smc_info.get("total_time", 0.0))
            # Store full SMC info (convert tensors to python types for serialization)
            smc_info_serializable = {
                k: (v.item() if hasattr(v, 'item') else v)
                for k, v in smc_info.items()
                if k not in ["ess_history", "accept_history", "temperature_history"]  # Skip large lists
            }
            history["smc_info"].append(smc_info_serializable)
        else:
            history["timing"]["smc_sampling"].append(0.0)
            history["smc_info"].append(None)

        # Fit new GP on updated data and compute metrics
        if model_builder_fn is not None:
            model = model_builder_fn(train_X, train_Y)
        else:
            model = build_and_fit_gp(
                train_X=train_X,
                train_Y=train_Y,
                input_bounds=bounds,
                kernel=kernel,
                hyperparameters=model_hyperparams,
                use_input_transform=use_input_transform,
                use_outcome_transform=use_outcome_transform,
                use_default_kernel_priors=use_default_kernel_priors,
                verbose=False,
            )

        # Store model hyperparameters
        model_hyperparam_dict = get_model_hyperparameters(model)
        history["model_hyperparameters"].append(model_hyperparam_dict)

        # Store updated data (skip for pool-based to save space - use selected_indices instead)
        if design_space_type != "pool":
            history["X"].append(train_X.clone())
            history["Y"].append(train_Y.clone())

        # Compute metrics after acquisition
        metric_start = time_module.perf_counter()
        if evaluation_grid is not None and len(performance_metrics) > 0:
            metrics_dict = _compute_metrics(
                model=model,
                evaluation_grid=evaluation_grid,
                eval_Y_true=eval_Y_true,
                metric_names=performance_metrics,
                density_fn=density_fn,
            )
            for metric_name, metric_value in metrics_dict.items():
                history["metrics"][metric_name].append(metric_value)

            if verbose:
                metric_strs = [f"{k}={v:.6f}" for k, v in metrics_dict.items()]
                print(f"  Metrics: {', '.join(metric_strs)}")
        metric_time = time_module.perf_counter() - metric_start

        # Record timing for this iteration
        iter_total_time = time_module.perf_counter() - iter_start_time
        history["timing"]["iteration_total"].append(iter_total_time)
        history["timing"]["gp_fitting"].append(gp_time)
        history["timing"]["acq_construction"].append(acq_construct_time)
        history["timing"]["acq_optimization"].append(acq_opt_time)
        history["timing"]["metric_computation"].append(metric_time)

        if verbose:
            smc_time = history["timing"]["smc_sampling"][-1]
            timing_str = f"  Timing: total={iter_total_time:.2f}s, gp={gp_time:.2f}s, acq_opt={acq_opt_time:.2f}s"
            if smc_time > 0:
                timing_str += f", smc={smc_time:.2f}s"
            print(timing_str)

        # Explicit garbage collection to prevent memory accumulation
        # GPyTorch creates lazy tensors and caches that may not be freed automatically
        del model
        del acq_function
        gc.collect()

    # Final summary
    history["final_X"] = train_X
    history["final_Y"] = train_Y

    if verbose:
        print("\n" + "=" * 60)
        print(f"Experiment completed! Final dataset size: {len(train_X)}")
        if "mse" in history["metrics"] and len(history["metrics"]["mse"]) > 0:
            print(f"Initial MSE: {history['metrics']['mse'][0]:.6f}")
            print(f"Final MSE: {history['metrics']['mse'][-1]:.6f}")
        # Print timing summary
        if len(history["timing"]["iteration_total"]) > 0:
            total_time = sum(history["timing"]["iteration_total"])
            avg_iter = total_time / len(history["timing"]["iteration_total"])
            avg_gp = sum(history["timing"]["gp_fitting"]) / len(history["timing"]["gp_fitting"])
            avg_acq_build = sum(history["timing"]["acq_construction"]) / len(history["timing"]["acq_construction"])
            avg_acq_opt = sum(history["timing"]["acq_optimization"]) / len(history["timing"]["acq_optimization"])
            print(f"Timing: total={total_time:.1f}s, avg_iter={avg_iter:.2f}s")
            print(f"  Avg breakdown: gp={avg_gp:.2f}s, acq_build={avg_acq_build:.2f}s, acq_opt={avg_acq_opt:.2f}s")

    return history


def _generate_initial_data(
    bounds: Tensor,
    n_samples: int,
    strategy: str = "random",
    seed: Optional[int] = None,
    candidates: Optional[Tensor] = None,
) -> Tensor:
    r"""Generate initial data using specified sampling strategy.

    Args:
        bounds: `(2, d)` tensor of bounds
        n_samples: Number of samples to generate
        strategy: Sampling strategy ("random", "sobol", "latin_hypercube")
        seed: Random seed
        candidates: Optional discrete candidates (if provided, sample from these)

    Returns:
        Tensor of shape `(n_samples, d)` with initial samples
    """
    dim = bounds.shape[1]

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # If discrete candidates provided, sample from them
    if candidates is not None:
        init_idx = torch.randperm(len(candidates))[:n_samples]
        return candidates[init_idx]

    # Continuous sampling
    if strategy == "random":
        X = torch.rand(n_samples, dim, dtype=bounds.dtype, device=bounds.device)

    elif strategy == "sobol":
        from torch.quasirandom import SobolEngine
        sobol_engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X = sobol_engine.draw(n_samples).to(dtype=bounds.dtype, device=bounds.device)

    elif strategy == "latin_hypercube":
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        X_np = sampler.random(n=n_samples)
        X = torch.tensor(X_np, dtype=bounds.dtype, device=bounds.device)

    else:
        raise ValueError(
            f"Unknown init_strategy: {strategy}. "
            "Supported: 'random', 'sobol', 'latin_hypercube'"
        )

    # Scale to bounds
    X = bounds[0] + X * (bounds[1] - bounds[0])
    return X


def _run_pool_experiment_from_config(config: Dict[str, Any], args) -> None:
    """Run a pool-based experiment from config (e.g., molecular AL).

    This handles the special case where we have a pre-defined pool of candidates
    instead of a continuous/grid search space.
    """
    from pathlib import Path
    from sidal.data_loaders import load_molecular_pool_data

    print("=" * 60)
    print("Pool-based Experiment")
    print("=" * 60)

    # Load molecular data
    pool_X, pool_Y, test_X, test_Y = load_molecular_pool_data(config, verbose=True)

    # Get experiment settings
    opt_config = config.get("optimization", {})
    n_init = config.get("initial_design", {}).get("n_init", 10)
    n_iterations = opt_config.get("n_iterations", 100)

    # Allow CLI override for n_iterations
    if args.n_iterations is not None:
        n_iterations = args.n_iterations
        print(f"CLI override: n_iterations={n_iterations}")

    # Get acquisition config
    acq_list = config.get("acquisition_functions", [])
    if args.acq_index is not None:
        if args.acq_index >= len(acq_list):
            raise ValueError(f"acq_index {args.acq_index} out of range (max {len(acq_list)-1})")
        acq_config = acq_list[args.acq_index]
    elif args.acq_name:
        matching = [a for a in acq_list if a["name"] == args.acq_name]
        if not matching:
            raise ValueError(f"Acquisition function '{args.acq_name}' not found in config")
        acq_config = matching[0]
    else:
        raise ValueError("Must specify --acq_name or --acq_index")

    acq_name = acq_config["name"]
    acq_hyperparams = acq_config.get("hyperparameters", {}).copy()  # Copy to allow override

    # If --lambda_val is specified, sync acq's internal lambda with eval lambda.
    if args.lambda_val is not None:
        apply_lambda_val_to_acq(acq_name, acq_hyperparams, args.lambda_val)
        print(f"CLI override: lambda_val={args.lambda_val} applied to {acq_name}")

    # Build model config
    model_cfg = config.get("model", {})
    model_config = {
        "kernel": model_cfg.get("kernel", "tanimoto"),
        "use_input_transform": model_cfg.get("use_input_transform", False),
        "use_outcome_transform": model_cfg.get("use_outcome_transform", True),
        "use_default_kernel_priors": model_cfg.get("use_default_kernel_priors", False),
        "noise_level": model_cfg.get("hyperparameters", {}).get("noise", 1e-4),
        "fix_noise": model_cfg.get("hyperparameters", {}).get("freeze_noise", True),
    }

    # Parse density function
    density_fn_str = config.get("density_fn_type", "lambda f: torch.exp(-f)")
    density_fn = eval(density_fn_str)

    acq_lambda = check_lambda_alignment(acq_name, acq_hyperparams, density_fn_str)
    if acq_lambda is not None:
        print(f"  Lambda alignment check passed ({acq_name}): effective lambda={acq_lambda}")

    # Get metrics
    metrics = config.get("metrics", {}).get("performance_metrics", ["mse"])

    # Dummy bounds for pool (not used but required by interface)
    dim = pool_X.shape[1]
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).double()

    print(f"\nExperiment: {acq_name}, seed={args.seed}")
    print(f"  n_init={n_init}, n_iterations={n_iterations}")
    print(f"  Pool: {len(pool_X)}, Test: {len(test_X)}, Dim: {dim}")

    # Prepare output directory
    output_config = config.get("output", {})
    output_dir = output_config.get("output_dir", "./exps/results/molecular")
    if args.output_dir is not None:
        output_dir = args.output_dir

    # Run experiment
    history = run_experiment(
        test_function=None,  # Not used for pool
        bounds=bounds,
        acq_name=acq_name,
        seed=args.seed,
        n_init=n_init,
        n_iterations=n_iterations,
        density_fn=density_fn,
        model_config=model_config,
        design_space_type="pool",
        acq_hyperparameters=acq_hyperparams,
        performance_metrics=metrics,
        verbose=True,
        # Pool-specific kwargs
        pool_X=pool_X,
        pool_Y=pool_Y,
        test_X=test_X,
        test_Y=test_Y,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    acq_suffix = ""
    if "constraint_type" in acq_hyperparams:
        ct = acq_hyperparams["constraint_type"]
        acq_suffix = "_EWV" if ct == "energy_weighted_variance" else "_NoC"

    filename = f"{acq_name}{acq_suffix}_seed{args.seed}.pt"
    filepath = output_path / filename

    torch.save(history, filepath)
    print(f"\nResults saved to: {filepath}")


def run_from_cli():
    """Command line interface for running experiments.

    Supports two modes:
    1. Config file mode: --config path/to/config.json --acq_name US --seed 0
    2. Direct mode: --problem Branin --acq_name US --seed 0 (for quick tests)
    """
    import argparse
    import json
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser(description="Run benchmarking experiment")

    # Config file mode
    parser.add_argument("--config", type=str, help="Path to JSON config file")

    # Required arguments
    parser.add_argument("--acq_name", type=str,
                       help="Acquisition function: RS, US, IMSE, HAL, AB-SID-iVAR, TS-SID-iVAR")
    parser.add_argument("--acq_index", type=int,
                       help="Index into acquisition_functions list (use instead of --acq_name for duplicates)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for DOE")

    # Direct mode arguments (used when --config not provided)
    parser.add_argument("--problem", type=str, default="Branin", help="Test function name")
    parser.add_argument("--n_init", type=int, default=5, help="Number of initial points")
    parser.add_argument("--n_iterations", type=int, default=None, help="Number of iterations (overrides config)")
    parser.add_argument("--noise_level", type=float, default=1e-4, help="Observation noise")
    parser.add_argument("--fix_noise", type=int, default=1, help="Fix noise (1=True, 0=False)")
    parser.add_argument("--domain", type=str, default="d", choices=["d", "c", "p"],
                       help="Domain type: 'd' (discrete/cartesian), 'c' (continuous), 'p' (pool)")
    parser.add_argument("--n_grid_per_dim", type=int, default=20, help="Grid points per dimension")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--acq_hyperparams", type=str, default="{}", help="JSON string of acq hyperparameters")
    parser.add_argument("--n_cpus", type=int, default=1, help="Number of CPUs per job")
    parser.add_argument("--override", type=str, action="append", default=[],
                       help="Override config values. Format: 'path.to.key=value'. "
                            "Can be used multiple times. Examples: "
                            "--override 'problem.dim=6' --override 'problem.kwargs.dim=6'")
    parser.add_argument("--lambda_val", type=float, default=None,
                       help="Boltzmann lambda value. Sets density_fn_type='lambda f: torch.exp(<lambda>*f)'")

    args = parser.parse_args()

    def apply_overrides(config, overrides):
        """Apply CLI overrides to config dict.

        Format: 'path.to.key=value' where value is auto-parsed as:
        - int if numeric without decimal
        - float if numeric with decimal
        - bool if 'true'/'false'
        - str otherwise

        Supports list indices: 'acquisition_functions.0.hyperparameters.key=value'
        """
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Invalid override format: {override}. Expected 'path.to.key=value'")
            key_path, value_str = override.split('=', 1)
            keys = key_path.strip().split('.')

            # Parse value
            value_str = value_str.strip()
            if value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            elif value_str.lower() == 'none':
                value = None
            elif value_str.startswith('[') or value_str.startswith('{'):
                # Parse as JSON array or object
                import json as json_module
                try:
                    value = json_module.loads(value_str)
                except json_module.JSONDecodeError:
                    value = value_str  # Keep as string if parsing fails
            else:
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str  # Keep as string

            # Navigate to parent and set value
            # Handle both dict keys and list indices
            obj = config
            for key in keys[:-1]:
                if key.isdigit():
                    obj = obj[int(key)]
                elif key not in obj:
                    obj[key] = {}
                    obj = obj[key]
                else:
                    obj = obj[key]

            # Set final value
            final_key = keys[-1]
            if final_key.isdigit():
                obj[int(final_key)] = value
            else:
                obj[final_key] = value
            print(f"Override: {key_path} = {value} ({type(value).__name__})")

        return config

    # Set CPU limits
    torch.set_num_threads(args.n_cpus)
    os.environ["OMP_NUM_THREADS"] = str(args.n_cpus)
    os.environ["MKL_NUM_THREADS"] = str(args.n_cpus)

    if args.config:
        # Config file mode - load all settings from JSON
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Apply CLI overrides to config
        if args.override:
            config = apply_overrides(config, args.override)

        # Apply lambda_val override if specified
        if args.lambda_val is not None:
            config["density_fn_type"] = density_fn_str_for_lambda(args.lambda_val)
            print(f"CLI override: density_fn_type = {config['density_fn_type']}")

        # Check if this is a pool-based experiment
        problem_type = config["problem"].get("type", "function")

        if problem_type == "pool":
            # Pool-based experiment (e.g., molecular AL)
            _run_pool_experiment_from_config(config, args)
            return

        # Get test function
        problem_name = config["problem"]["name"]
        problem_kwargs = config["problem"].get("kwargs", {})
        # Include dim if specified at top level (for Hartmann, etc.)
        # Note: Some functions (Ishigami, Branin, etc.) have fixed dim and don't accept this arg
        problem_dim = config["problem"].get("dim", None)
        if problem_dim is not None and "dim" not in problem_kwargs:
            problem_kwargs["dim"] = problem_dim

        from sidal.test_functions import benchmark_test_functions as test_functions
        if problem_name in ("GPSample", "GPSampleWithBias"):
            # Use GPSampleWithBias for both - it's a superset of GPSample
            # GPSample configs work with use_bias=False (default)
            from sidal.test_functions import GPSampleWithBias

            gp_seed = problem_kwargs.get("seed", 0)
            dim = problem_kwargs.get("dim", 2)
            kernel = problem_kwargs.get("kernel", "matern52")
            lengthscale = problem_kwargs.get("lengthscale", [0.3] * dim)
            outputscale = problem_kwargs.get("outputscale", 1.0)
            n_features = problem_kwargs.get("n_features", 2048)
            use_bias = problem_kwargs.get("use_bias", False)
            bias_lengthscale = problem_kwargs.get("bias_lengthscale", lengthscale)
            bias_outputscale = problem_kwargs.get("bias_outputscale", 1.0)
            bias_seed = problem_kwargs.get("bias_seed", 1)

            test_fn = GPSampleWithBias(
                dim=dim,
                kernel=kernel,
                lengthscale=lengthscale,
                outputscale=outputscale,
                n_features=n_features,
                seed=gp_seed,
                use_bias=use_bias,
                bias_lengthscale=bias_lengthscale,
                bias_outputscale=bias_outputscale,
                bias_seed=bias_seed,
            )
            print(f"Created GPSampleWithBias with seed={gp_seed}, use_bias={use_bias}" +
                  (f", bias_seed={bias_seed}" if use_bias else ""))
        elif hasattr(test_functions, problem_name):
            test_fn_class = getattr(test_functions, problem_name)
            # Remove 'seed' from kwargs if present (not used by standard test functions)
            problem_kwargs_clean = {k: v for k, v in problem_kwargs.items() if k != 'seed'}
            # Try with dim first, fall back to without if function has fixed dimension
            try:
                test_fn = test_fn_class(**problem_kwargs_clean)
            except TypeError:
                problem_kwargs_clean.pop('dim', None)
                test_fn = test_fn_class(**problem_kwargs_clean)
        else:
            # Check custom test functions (Forrester, Gramacy1D, HOnCu13, SiSWEnergy, etc.)
            # Note: GPSample and GPSampleWithBias are handled above
            from sidal.test_functions import (
                Forrester,
                H2CuMACEPES,
                Gramacy1D,
                H2OonPt111_Slab,
                HOnCu13,
                Higdon,
                SiSWEnergy,
            )
            custom_test_functions = {
                "Forrester": Forrester,
                "H2CuMACEPES": H2CuMACEPES,
                "Gramacy1D": Gramacy1D,
                "H2OonPt111_Slab": H2OonPt111_Slab,
                "HOnCu13": HOnCu13,
                "Higdon": Higdon,
                "SiSWEnergy": SiSWEnergy,
            }
            if problem_name in custom_test_functions:
                test_fn_class = custom_test_functions[problem_name]
                problem_kwargs_clean = {k: v for k, v in problem_kwargs.items() if k != 'seed'}
                # Try with dim first, fall back to without if function has fixed dimension
                try:
                    test_fn = test_fn_class(**problem_kwargs_clean)
                except TypeError:
                    problem_kwargs_clean.pop('dim', None)
                    test_fn = test_fn_class(**problem_kwargs_clean)
            else:
                raise ValueError(f"Unknown problem: {problem_name}")

        # Get model settings
        model_config = config.get("model", {})
        model_hyperparams = model_config.get("hyperparameters", {})
        noise_level = model_hyperparams.get("noise", 1e-4)
        fix_noise = model_hyperparams.get("freeze_noise", True)
        kernel = model_config.get("kernel", "matern52")
        use_input_transform = model_config.get("use_input_transform", True)
        use_outcome_transform = model_config.get("use_outcome_transform", True)
        use_default_kernel_priors = model_config.get("use_default_kernel_priors", False)

        # Handle fixed hyperparameters (hyperparameter_mode: "fixed")
        hyperparameter_mode = model_config.get("hyperparameter_mode", "learned")
        if hyperparameter_mode == "fixed":
            fixed_lengthscale = model_hyperparams.get("lengthscale", None)
            fixed_outputscale = model_hyperparams.get("outputscale", None)
        else:
            fixed_lengthscale = None
            fixed_outputscale = None

        # Get experiment settings
        opt_config = config.get("optimization", {})
        n_init = config.get("initial_design", {}).get("n_init", 10)
        n_iterations = opt_config.get("n_iterations", 50)
        # Map domain flag to design_space_type
        domain_map = {"d": "discrete", "c": "continuous", "p": "pool"}
        design_space_type = domain_map[args.domain]

        # Compute n_grid_per_dim: either directly specified or derived from max_grid_points
        dim = test_fn.bounds.shape[1]
        if "n_grid_per_dim" in opt_config:
            n_grid_per_dim = opt_config["n_grid_per_dim"]
        elif "max_grid_points" in opt_config:
            # Compute n_grid_per_dim = floor(max_grid_points^(1/d))
            max_grid_points = opt_config["max_grid_points"]
            n_grid_per_dim = int(max_grid_points ** (1.0 / dim))
            actual_grid_points = n_grid_per_dim ** dim
            print(f"Grid budget: max={max_grid_points}, n_grid_per_dim={n_grid_per_dim}, actual={actual_grid_points}")
        else:
            n_grid_per_dim = 20  # Default

        # Allow CLI overrides for n_iterations (useful for smoke tests)
        if args.n_iterations is not None:
            n_iterations = args.n_iterations
            print(f"CLI override: n_iterations={n_iterations}")

        # Get continuous optimization parameters
        num_restarts = opt_config.get("num_restarts", 20)
        raw_samples = opt_config.get("raw_samples", 1024)

        # Get acquisition config - either by index or by name
        acq_list = config.get("acquisition_functions", [])
        if args.acq_index is not None:
            # Use index to support duplicate names with different hyperparameters
            if args.acq_index < 0 or args.acq_index >= len(acq_list):
                raise ValueError(f"acq_index {args.acq_index} out of range [0, {len(acq_list)-1}]")
            acq_config = acq_list[args.acq_index]
            args.acq_name = acq_config["name"]  # Set acq_name from config
            acq_hyperparams = acq_config.get("hyperparameters", {})
        elif args.acq_name:
            # Find by name (legacy behavior - gets first match)
            acq_hyperparams = {}
            for acq_config in acq_list:
                if acq_config["name"] == args.acq_name:
                    acq_hyperparams = acq_config.get("hyperparameters", {})
                    break
        else:
            raise ValueError("Either --acq_name or --acq_index is required")

        # If --lambda_val is specified, sync acq's internal lambda with eval lambda.
        if args.lambda_val is not None:
            acq_hyperparams = acq_hyperparams.copy()
            apply_lambda_val_to_acq(args.acq_name, acq_hyperparams, args.lambda_val)
            print(f"CLI override: lambda_val={args.lambda_val} applied to {args.acq_name}")

        # Get output settings - allow CLI override
        # If CLI provides --output_dir, use it as-is (shell script already added domain folder)
        # Otherwise, insert domain folder into config's output_dir
        if args.output_dir is not None:
            output_dir = args.output_dir
            print(f"CLI override: output_dir={output_dir}")
        else:
            output_dir = config.get("output", {}).get("output_dir", "./results")
            # Insert domain folder (discrete/continuous/pool) into output path
            # e.g., ./results/formal_benchmark/Branin_2d/matern52
            #    -> ./results/formal_benchmark/discrete/Branin_2d/matern52
            domain_folder = design_space_type
            output_path = Path(output_dir)
            parts = list(output_path.parts)
            # Insert domain folder before the last 2 components (problem/kernel)
            if len(parts) >= 2:
                parts.insert(-2, domain_folder)
                output_dir = str(Path(*parts))
            else:
                output_dir = str(output_path / domain_folder)

        # Get metrics config
        metrics_config = config.get("metrics", {})
        metrics = metrics_config.get("performance_metrics", ["mse"])
        n_eval_points = metrics_config.get("n_eval_points", 1000)
        eval_seed = metrics_config.get("eval_seed", 42)

        # Load precomputed evaluation grid if specified
        eval_grid_path = metrics_config.get("eval_grid_path", None)
        precomputed_eval_grid = None
        precomputed_eval_Y = None
        if eval_grid_path is not None:
            eval_grid_data = torch.load(eval_grid_path, map_location="cpu", weights_only=False)
            precomputed_eval_grid = eval_grid_data["eval_X"]
            precomputed_eval_Y = eval_grid_data["eval_Y"]
            print(f"Loaded precomputed eval grid: {len(precomputed_eval_grid)} points from {eval_grid_path}")

        # Get output config (including save_reference_points)
        output_config = config.get("output", {})
        save_reference_points = output_config.get("save_reference_points", False)

        # Parse density_fn from config (for weighted metrics)
        density_fn = None
        density_fn_type = config.get("density_fn_type", None)

        # Lambda alignment check (continuous path) - same helper as pool-mode.
        if density_fn_type is not None:
            _aligned = check_lambda_alignment(args.acq_name, acq_hyperparams, density_fn_type)
            if _aligned is not None:
                print(f"  Lambda alignment check passed ({args.acq_name}): effective lambda={_aligned}")

        if density_fn_type is not None:
            # Convert string like "lambda f: torch.exp(-f)" to callable
            # Security note: eval is used here but config files are trusted
            try:
                density_fn = eval(density_fn_type)
                if not callable(density_fn):
                    print(f"Warning: density_fn_type '{density_fn_type}' is not callable, ignoring")
                    density_fn = None
                else:
                    print(f"Using density_fn: {density_fn_type}")
            except Exception as e:
                print(f"Warning: Failed to parse density_fn_type '{density_fn_type}': {e}")
                density_fn = None

    else:
        # Direct mode - use command line arguments
        from sidal.test_functions import benchmark_test_functions as test_functions
        if hasattr(test_functions, args.problem):
            test_fn = getattr(test_functions, args.problem)()
        else:
            raise ValueError(f"Unknown problem: {args.problem}")

        n_init = args.n_init
        n_iterations = args.n_iterations if args.n_iterations is not None else 20
        acq_hyperparams = json.loads(args.acq_hyperparams) if args.acq_hyperparams else None
        output_dir = args.output_dir if args.output_dir is not None else "./results"
        metrics = ["mse"]
        # Map domain flag to design_space_type
        domain_map = {"d": "discrete", "c": "continuous", "p": "pool"}
        design_space_type = domain_map[args.domain]
        n_grid_per_dim = args.n_grid_per_dim
        n_eval_points = 1000  # Default for direct mode
        eval_seed = 42  # Default for direct mode
        save_reference_points = False  # Default for direct mode
        num_restarts = 20  # Default for direct mode
        raw_samples = 1024  # Default for direct mode
        # Default density function for direct mode
        density_fn = lambda f: torch.exp(-f)

        # Add domain folder to output path
        output_dir = str(Path(output_dir) / design_space_type)

        # Build model_config for direct mode
        model_config = {
            "kernel": "matern52",
            "use_input_transform": True,
            "use_outcome_transform": True,
            "use_default_kernel_priors": False,
            "noise_level": args.noise_level,
            "fix_noise": bool(args.fix_noise),
        }

    # Build model_config for config file mode (if not already built in direct mode)
    if args.config:
        model_config = {
            "kernel": kernel,
            "use_input_transform": use_input_transform,
            "use_outcome_transform": use_outcome_transform,
            "use_default_kernel_priors": use_default_kernel_priors,
            "noise_level": noise_level,
            "fix_noise": fix_noise,
        }
        if fixed_lengthscale is not None:
            model_config["fixed_lengthscale"] = fixed_lengthscale
        if fixed_outputscale is not None:
            model_config["fixed_outputscale"] = fixed_outputscale

        # Validate density_fn is set (required for proper benchmarking)
        if density_fn is None:
            raise ValueError(
                "density_fn is required. Please set 'density_fn_type' in config file "
                "(e.g., \"density_fn_type\": \"lambda f: torch.exp(-f)\")"
            )

    # Run experiment with DOE seed (args.seed varies per repetition)
    # Check if HAL - use separate runner since HAL doesn't fit standard acquisition pattern
    if args.acq_name.upper() == "HAL":
        if design_space_type != "continuous":
            print(f"WARNING: HAL is only supported for continuous domains. Skipping.")
            print(f"  (Use --domain c to run HAL)")
            return  # Skip silently for non-continuous experiments
        history = run_hal_experiment(
            test_function=test_fn,
            bounds=test_fn.bounds,
            seed=args.seed,
            n_init=n_init,
            n_iterations=n_iterations,
            density_fn=density_fn,
            model_config=model_config,
            hal_hyperparameters=acq_hyperparams,
            performance_metrics=metrics,
            evaluation_grid=precomputed_eval_grid,
            n_eval_points=n_eval_points,
            eval_seed=eval_seed,
            verbose=True,
            precomputed_eval_Y=precomputed_eval_Y,
        )
    else:
        history = run_experiment(
            test_function=test_fn,
            bounds=test_fn.bounds,
            acq_name=args.acq_name,
            seed=args.seed,  # This controls DOE, different per repetition
            n_init=n_init,
            n_iterations=n_iterations,
            density_fn=density_fn,
            model_config=model_config,
            design_space_type=design_space_type,
            n_grid_per_dim=n_grid_per_dim,
            acq_hyperparameters=acq_hyperparams,
            performance_metrics=metrics,
            evaluation_grid=precomputed_eval_grid,
            n_eval_points=n_eval_points,
            eval_seed=eval_seed,
            save_reference_points=save_reference_points,
            verbose=True,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            precomputed_eval_Y=precomputed_eval_Y,
        )

    # Store RFF params for GPSample (for reconstruction in visualization)
    if hasattr(test_fn, 'get_rff_params'):
        history["rff_params"] = test_fn.get_rff_params()

    # Store problem info for visualization
    history["problem_name"] = config["problem"]["name"]
    history["problem_kwargs"] = config["problem"].get("kwargs", {})

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build filename with hyperparameters using short abbreviations
    # Format: {base_name}_{constraint_abbrev}_{energy_abbrev}_seed{Y}
    filename_parts = [args.acq_name]

    # Abbreviation mappings
    constraint_abbrevs = {
        "energy_weighted_variance": "EWV",  # Energy Weighted Variance
    }
    base_approx_abbrevs = {
        "mean": "Mean",
    }

    # Add constraint_type abbreviation if specified
    if "constraint_type" in acq_hyperparams:
        ct = acq_hyperparams["constraint_type"]
        if ct is None:
            filename_parts.append("NoC")  # No Constraint
        else:
            abbrev = constraint_abbrevs.get(ct, ct)
            filename_parts.append(abbrev)

    # Add base_approximation abbreviation if specified
    if "base_approximation" in acq_hyperparams:
        ba = acq_hyperparams["base_approximation"]
        if ba is None:
            filename_parts.append("NoE")  # No Energy term
        else:
            abbrev = base_approx_abbrevs.get(ba, ba)
            filename_parts.append(abbrev)

    # Add thompson sampling indicator if True
    if acq_hyperparams.get("use_thompson_sampling", False):
        filename_parts.append("TS")

    filename_parts.append(f"seed{args.seed}")
    filename = "_".join(filename_parts) + ".pt"

    torch.save(history, output_path / filename)
    print(f"\nSaved to: {output_path / filename}")


def _compute_metrics(
    model: Any,
    evaluation_grid: Tensor,
    eval_Y_true: Tensor,
    metric_names: List[str],
    density_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Dict[str, float]:
    r"""Compute performance metrics for current model.

    Args:
        model: Fitted GP model
        evaluation_grid: Tensor of shape `(n_eval, d)` for evaluation
        eval_Y_true: True function values at evaluation points `(n_eval,)`
        metric_names: List of metrics to compute
        density_fn: Optional density function for weighted metrics

    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}

    with torch.no_grad():
        posterior = model.posterior(evaluation_grid)
        pred_mean = posterior.mean.squeeze()
        pred_variance = posterior.variance.squeeze()

        errors = pred_mean - eval_Y_true
        squared_errors = errors ** 2

        if "mse" in metric_names:
            metrics["mse"] = squared_errors.mean().item()

        if "nlml" in metric_names:
            log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
            nlpl_per_point = 0.5 * torch.log(pred_variance) + 0.5 * squared_errors / pred_variance + 0.5 * log_2pi
            metrics["nlml"] = nlpl_per_point.mean().item()

        if "weighted_mse" in metric_names:
            if density_fn is not None and callable(density_fn):
                density_weights = density_fn(eval_Y_true)
                density_weights = density_weights / density_weights.sum()
                metrics["weighted_mse"] = (squared_errors * density_weights).sum().item()
            else:
                # Don't silently fall back to MSE - use NaN to indicate missing density_fn
                metrics["weighted_mse"] = float('nan')

        if "weighted_nlml" in metric_names:
            log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
            nlpl_per_point = 0.5 * torch.log(pred_variance) + 0.5 * squared_errors / pred_variance + 0.5 * log_2pi
            if density_fn is not None and callable(density_fn):
                density_weights = density_fn(eval_Y_true)
                density_weights = density_weights / density_weights.sum()
                metrics["weighted_nlml"] = (nlpl_per_point * density_weights).sum().item()
            else:
                # Don't silently fall back to NLML - use NaN to indicate missing density_fn
                metrics["weighted_nlml"] = float('nan')

        if "mae" in metric_names:
            metrics["mae"] = errors.abs().mean().item()

        if "max_error" in metric_names:
            metrics["max_error"] = errors.abs().max().item()

        # Explicit cleanup of posterior and intermediate tensors
        del posterior, pred_mean, pred_variance, errors, squared_errors

    return metrics


def run_hal_experiment(
    test_function: Union[BaseTestProblem, Callable[[Tensor], Tensor]],
    bounds: Tensor,
    seed: int,
    n_init: int,
    n_iterations: int,
    density_fn: Callable[[Tensor], Tensor],
    model_config: Dict[str, Any],
    hal_hyperparameters: Optional[Dict[str, Any]] = None,
    init_strategy: str = "random",
    performance_metrics: Optional[List[str]] = None,
    evaluation_grid: Optional[Tensor] = None,
    n_eval_points: int = 1000,
    eval_seed: int = 42,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    r"""Run HAL-style active learning experiment.

    Implements the Hyperactive Learning (HAL) approach:
    1. Run single-chain MCMC from biased distribution P_HAL(x) ∝ exp(-(μ - τσ))
    2. At each step, check if relative uncertainty σ/|μ| exceeds threshold
    3. If threshold exceeded → STOP and return that point (reactive selection)

    This is fundamentally different from standard acquisition optimization:
    - Standard: Compute acq(x) everywhere, pick argmax
    - HAL: Sample from biased distribution, apply threshold criterion

    Reference: van der Oord et al., "Hyperactive learning for data-driven
    interatomic potentials", npj Computational Materials (2023)

    Args:
        test_function: Either a BoTorch BaseTestProblem or a callable.
        bounds: A `(2, d)` tensor of lower and upper bounds.
        seed: Random seed for reproducibility.
        n_init: Number of initial random samples.
        n_iterations: Number of sequential acquisition iterations.
        density_fn: Density function for weighted metrics.
        model_config: Dictionary of GP model configuration.
        hal_hyperparameters: HAL-specific hyperparameters:
            - tau: Exploration parameter in E_HAL = μ - τσ (default: 1.0).
                Can be a [min, max] list to sample uniformly per experiment.
            - temperature: Temperature for Boltzmann distribution (default: 1.0)
            - threshold: Uncertainty threshold for stopping (default: 0.1).
                Can be a [min, max] list to sample uniformly per experiment.
            - max_mcmc_steps: Max MCMC steps per acquisition (default: 1000)
            - burn_in_steps: Steps before threshold checking (default: 50)
            - step_size: MCMC step size as fraction of domain (default: 0.1)
            - init_strategy: "from_data" (default) or "random"
        init_strategy: Initial sampling strategy.
        performance_metrics: List of metrics to compute.
        evaluation_grid: Optional evaluation points.
        n_eval_points: Number of evaluation points if grid not provided.
        eval_seed: Random seed for evaluation grid.
        verbose: If True, print progress.

    Returns:
        History dictionary with experiment results.
    """
    from sidal import HAL_Acquisition
    import time as time_module

    # Validate model_config
    required_model_keys = ["kernel", "use_input_transform", "use_outcome_transform",
                          "use_default_kernel_priors", "noise_level", "fix_noise"]
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"model_config must contain '{key}'")

    # Default HAL hyperparameters
    if hal_hyperparameters is None:
        hal_hyperparameters = {}

    # tau and threshold can be either:
    # - A single value (used directly)
    # - A list/tuple [min, max] (sample uniformly for this experiment)
    tau_config = hal_hyperparameters.get("tau", 1.0)
    threshold_config = hal_hyperparameters.get("threshold", 0.1)

    # Sample tau and threshold from ranges if specified
    rng = np.random.RandomState(seed)
    if isinstance(tau_config, (list, tuple)) and len(tau_config) == 2:
        tau = rng.uniform(tau_config[0], tau_config[1])
    else:
        tau = tau_config

    if isinstance(threshold_config, (list, tuple)) and len(threshold_config) == 2:
        threshold = rng.uniform(threshold_config[0], threshold_config[1])
    else:
        threshold = threshold_config

    max_mcmc_steps = hal_hyperparameters.get("max_mcmc_steps", 1000)
    burn_in_steps = hal_hyperparameters.get("burn_in_steps", 50)
    step_size = hal_hyperparameters.get("step_size", 0.1)
    mcmc_init = hal_hyperparameters.get("init_strategy", "from_data")

    # Convert boltzmann_lambda to temperature + boltzmann_sign (same as AB-SID-iVAR)
    if "boltzmann_lambda" in hal_hyperparameters:
        bl = hal_hyperparameters["boltzmann_lambda"]
        temperature = 1.0 / abs(bl)
        boltzmann_sign = 1.0 if bl > 0 else -1.0
    else:
        temperature = hal_hyperparameters.get("temperature", 1.0)
        boltzmann_sign = hal_hyperparameters.get("boltzmann_sign", -1.0)

    # Extract model config
    kernel = model_config["kernel"]
    use_input_transform = model_config["use_input_transform"]
    use_outcome_transform = model_config["use_outcome_transform"]
    use_default_kernel_priors = model_config["use_default_kernel_priors"]
    noise_level = model_config["noise_level"]
    fix_noise = model_config["fix_noise"]
    fixed_lengthscale = model_config.get("fixed_lengthscale", None)
    fixed_outputscale = model_config.get("fixed_outputscale", None)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = bounds.shape[1]
    dtype = bounds.dtype
    device = bounds.device

    # Prepare evaluation function
    if isinstance(test_function, BaseTestProblem):
        def eval_fn(X: Tensor) -> Tensor:
            y = test_function(X)
            return y.unsqueeze(-1) if y.dim() == 1 else y
    else:
        def eval_fn(X: Tensor) -> Tensor:
            y = test_function(X)
            return y.unsqueeze(-1) if y.dim() == 1 else y

    # Generate evaluation grid for metrics
    if performance_metrics is None:
        performance_metrics = ["mse"]

    if evaluation_grid is None and len(performance_metrics) > 0:
        torch.manual_seed(eval_seed)
        evaluation_grid = torch.rand(n_eval_points, dim, dtype=dtype, device=device)
        evaluation_grid = bounds[0] + evaluation_grid * (bounds[1] - bounds[0])
        if verbose:
            print(f"Evaluation grid: {len(evaluation_grid)} random points (eval_seed={eval_seed})")

    # Precompute true function values on evaluation grid
    eval_Y_true = None
    if evaluation_grid is not None:
        if kwargs.get("precomputed_eval_Y") is not None:
            eval_Y_true = kwargs["precomputed_eval_Y"].squeeze()
            if verbose:
                print(f"Using precomputed eval_Y_true ({len(eval_Y_true)} points)")
        else:
            # Ground-truth evaluation must not consume the experiment's random
            # stream (e.g. MACE draws from torch's global RNG); otherwise the
            # observation noise of every later query depends on it.
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                eval_Y_true = eval_fn(evaluation_grid).squeeze()

    # Generate initial data
    train_X = _generate_initial_data(bounds, n_init, init_strategy, seed, candidates=None)
    train_Y = eval_fn(train_X)

    # Add noise to observations
    if noise_level > 0:
        train_Y = train_Y + torch.randn_like(train_Y) * (noise_level ** 0.5)

    # Initialize history
    history = {
        "X": [train_X.clone()],
        "Y": [train_Y.clone()],
        "candidates_selected": [],
        "acq_values": [],  # Empty for HAL (no acquisition value optimization)
        "reference_points": [],  # Empty for HAL (no reference points)
        "hal_info": [],  # HAL-specific selection info
        "model_hyperparameters": [],
        "metrics": defaultdict(list),
        "timing": {
            "iteration_total": [],
            "gp_fitting": [],
            "hal_selection": [],
            "acq_construction": [],  # Empty for HAL
            "acq_optimization": [],  # Empty for HAL
            "metric_computation": [],
            "smc_sampling": [],  # Empty for HAL
        },
        "smc_info": [],  # Empty for HAL
        "config": {
            "acq_name": "HAL",
            "seed": seed,
            "n_init": n_init,
            "n_iterations": n_iterations,
            "dim": dim,
            "bounds": bounds.tolist(),
        },
        "model_config": model_config,
        "hal_hyperparameters": hal_hyperparameters,
        "hal_sampled_params": {
            "tau": tau,
            "threshold": threshold,
            "boltzmann_sign": boltzmann_sign,
            "temperature": temperature,
        },
    }

    # Build model hyperparameters dict
    model_hyperparams = {
        "noise": noise_level,
        "freeze_noise": fix_noise,
    }
    if fixed_lengthscale is not None:
        model_hyperparams["lengthscale"] = fixed_lengthscale
        model_hyperparams["freeze_lengthscale"] = True
    if fixed_outputscale is not None:
        model_hyperparams["outputscale"] = fixed_outputscale
        model_hyperparams["freeze_outputscale"] = True

    if verbose:
        print(f"\nStarting HAL experiment")
        tau_info = f"tau: {tau:.4f}" + (f" (sampled from {tau_config})" if isinstance(tau_config, (list, tuple)) else "")
        threshold_info = f"threshold: {threshold:.4f}" + (f" (sampled from {threshold_config})" if isinstance(threshold_config, (list, tuple)) else "")
        print(f"  Seed: {seed}, {tau_info}, {threshold_info}")
        print(f"  max_mcmc_steps: {max_mcmc_steps}, step_size: {step_size}, mcmc_init: {mcmc_init}")
        print(f"  Noise: {noise_level}, Init: {n_init}, Iterations: {n_iterations}")
        print("=" * 60)

    from .model_builder import get_model_hyperparameters

    # Compute initial metrics
    if evaluation_grid is not None and len(performance_metrics) > 0:
        initial_model = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=bounds,
            kernel=kernel,
            hyperparameters=model_hyperparams,
            use_input_transform=use_input_transform,
            use_outcome_transform=use_outcome_transform,
            use_default_kernel_priors=use_default_kernel_priors,
            verbose=False,
        )
        history["model_hyperparameters"].append(get_model_hyperparameters(initial_model))

        metrics_dict = _compute_metrics(
            model=initial_model,
            evaluation_grid=evaluation_grid,
            eval_Y_true=eval_Y_true,
            metric_names=performance_metrics,
            density_fn=density_fn,
        )
        for metric_name, metric_value in metrics_dict.items():
            history["metrics"][metric_name].append(metric_value)

        if verbose:
            metric_strs = [f"{k}={v:.6f}" for k, v in metrics_dict.items()]
            print(f"Initial metrics (iter 0): {', '.join(metric_strs)}")

    # Main HAL loop
    for iteration in range(n_iterations):
        iter_start_time = time_module.perf_counter()
        if verbose:
            print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Build and fit GP
        gp_start_time = time_module.perf_counter()
        model = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=bounds,
            kernel=kernel,
            hyperparameters=model_hyperparams,
            use_input_transform=use_input_transform,
            use_outcome_transform=use_outcome_transform,
            use_default_kernel_priors=use_default_kernel_priors,
            verbose=False,
        )
        gp_time = time_module.perf_counter() - gp_start_time

        # HAL selection: single-chain MCMC with threshold stopping
        hal_start_time = time_module.perf_counter()
        hal_acq = HAL_Acquisition(
            model=model,
            bounds=bounds,
            tau=tau,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            threshold=threshold,
            max_mcmc_steps=max_mcmc_steps,
            burn_in_steps=burn_in_steps,
            step_size=step_size,
            seed=seed + iteration,  # Different seed per iteration
            train_X=train_X,
            init_strategy=mcmc_init,
        )
        new_x, hal_info = hal_acq.select_next_point()
        hal_time = time_module.perf_counter() - hal_start_time

        # Ensure new_x has shape (1, d)
        if new_x.dim() > 2:
            new_x = new_x.squeeze()
        if new_x.dim() == 1:
            new_x = new_x.unsqueeze(0)

        # Evaluate test function
        new_y = eval_fn(new_x)

        # Add noise
        if noise_level > 0:
            new_y = new_y + torch.randn_like(new_y) * (noise_level ** 0.5)

        if verbose:
            x_str = ", ".join([f"{new_x[0, i].item():.4f}" for i in range(dim)])
            print(f"  Selected: [{x_str}], Value: {new_y.item():.4f}")
            print(f"  HAL: method={hal_info['selection_method']}, "
                  f"steps={hal_info['steps_taken']}, "
                  f"sigma={hal_info['selected_sigma']:.4f}")

        # Update training data
        train_X = torch.cat([train_X, new_x], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)

        # Record
        history["candidates_selected"].append(new_x.clone())
        history["hal_info"].append(hal_info)
        history["acq_values"].append(None)  # Placeholder for compatibility
        history["reference_points"].append(None)  # Placeholder for compatibility
        history["smc_info"].append(None)  # Placeholder for compatibility

        # Fit new GP and compute metrics
        model = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=bounds,
            kernel=kernel,
            hyperparameters=model_hyperparams,
            use_input_transform=use_input_transform,
            use_outcome_transform=use_outcome_transform,
            use_default_kernel_priors=use_default_kernel_priors,
            verbose=False,
        )
        history["model_hyperparameters"].append(get_model_hyperparameters(model))
        history["X"].append(train_X.clone())
        history["Y"].append(train_Y.clone())

        # Compute metrics
        metric_start = time_module.perf_counter()
        if evaluation_grid is not None and len(performance_metrics) > 0:
            metrics_dict = _compute_metrics(
                model=model,
                evaluation_grid=evaluation_grid,
                eval_Y_true=eval_Y_true,
                metric_names=performance_metrics,
                density_fn=density_fn,
            )
            for metric_name, metric_value in metrics_dict.items():
                history["metrics"][metric_name].append(metric_value)

            if verbose:
                metric_strs = [f"{k}={v:.6f}" for k, v in metrics_dict.items()]
                print(f"  Metrics: {', '.join(metric_strs)}")
        metric_time = time_module.perf_counter() - metric_start

        # Record timing
        iter_total_time = time_module.perf_counter() - iter_start_time
        history["timing"]["iteration_total"].append(iter_total_time)
        history["timing"]["gp_fitting"].append(gp_time)
        history["timing"]["hal_selection"].append(hal_time)
        history["timing"]["acq_construction"].append(0.0)  # Placeholder for compatibility
        history["timing"]["acq_optimization"].append(0.0)  # Placeholder for compatibility
        history["timing"]["smc_sampling"].append(0.0)  # Placeholder for compatibility
        history["timing"]["metric_computation"].append(metric_time)

        if verbose:
            print(f"  Timing: total={iter_total_time:.2f}s, gp={gp_time:.2f}s, hal={hal_time:.2f}s")

        # Explicit garbage collection to prevent memory accumulation
        del model
        del hal_acq
        gc.collect()

    # Final summary
    history["final_X"] = train_X
    history["final_Y"] = train_Y

    if verbose:
        print("\n" + "=" * 60)
        print(f"HAL experiment completed! Final dataset size: {len(train_X)}")
        if "mse" in history["metrics"] and len(history["metrics"]["mse"]) > 0:
            print(f"Initial MSE: {history['metrics']['mse'][0]:.6f}")
            print(f"Final MSE: {history['metrics']['mse'][-1]:.6f}")

        # HAL-specific summary
        n_threshold = sum(1 for info in history["hal_info"] if info.get("triggered_threshold", False))
        print(f"Threshold triggered: {n_threshold}/{n_iterations} iterations")

    return history


if __name__ == "__main__":
    run_from_cli()
