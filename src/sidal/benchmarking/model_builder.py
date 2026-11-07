#!/usr/bin/env python3
r"""
Model builders for benchmarking active learning and Bayesian optimization.

This module provides utilities for building and training GP models with
configurable kernels and hyperparameters.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from torch import Tensor


def build_and_fit_gp(
    train_X: Tensor,
    train_Y: Tensor,
    input_bounds: Tensor,
    kernel: str = "matern52",
    hyperparameters: Optional[Dict[str, Any]] = None,
    fit_model: bool = True,
    max_retries: int = 3,
    verbose: bool = False,
    use_input_transform: bool = False,
    use_outcome_transform: Optional[bool] = None,
    use_default_kernel_priors: bool = False,
) -> SingleTaskGP:
    r"""Build and fit a SingleTaskGP model with specified kernel and hyperparameters.

    This function extracts the model building and training logic commonly used in
    active learning benchmarks, allowing for:
    - Flexible kernel specification
    - Optional fixed hyperparameters (for controlled experiments)
    - Consistent model training across different experiments

    Args:
        train_X: Training inputs of shape `(n, d)` where n is the number of
            observations and d is the input dimension.
        train_Y: Training targets of shape `(n, 1)` or `(n,)`. If 1-dimensional,
            will be reshaped to `(n, 1)`.
        input_bounds: A `(2, d)` tensor specifying the lower and upper bounds for each
            input dimension. This is REQUIRED for benchmarking to ensure consistent
            input normalization across iterations. The first row contains lower bounds,
            the second row contains upper bounds. When `use_input_transform=True`,
            inputs are normalized to [0, 1] using these fixed bounds (not learned from data).
        kernel: Kernel type specification. Supported options:
            - "rbf": RBF (Gaussian) kernel with ScaleKernel
            - "matern": Matern kernel with nu=2.5 (default) and ScaleKernel
            - "matern52": Matern kernel with nu=2.5
            - "matern32": Matern kernel with nu=1.5
            - "matern12": Matern kernel with nu=0.5
            - "tanimoto": Tanimoto kernel for molecular fingerprints (requires gauche)
        hyperparameters: Optional dictionary of hyperparameters to fix. Supported keys:
            - "lengthscale": float or Tensor of shape `(d,)` for ARD
            - "outputscale": float for the output scale (only for ScaleKernel)
            - "noise": float for observation noise variance
            - "freeze_lengthscale": bool, if True, freeze lengthscale during training
            - "freeze_outputscale": bool, if True, freeze outputscale during training
            - "freeze_noise": bool, if True, freeze noise during training
            If None, all hyperparameters will be learned via MLE.
        fit_model: If True, fit the model using exact marginal log likelihood.
            If False, return the model without fitting (useful for custom training).
        max_retries: Maximum number of retries for fitting if it fails. Only used
            if fit_model=True.
        verbose: If True, print hyperparameter values after fitting.
        use_input_transform: If True, apply Normalize transform to inputs using the
            provided `input_bounds` (normalizes to [0, 1] cube). If False (default),
            no input transform is applied and the GP works in the original input space.
        use_outcome_transform: If True, apply Standardize transform to outputs (zero mean, unit variance).
            If False, no outcome transform. If None, raises an error (must be explicitly set).
            Recommended: True for learned hyperparameters, False for GP prior benchmarks.
        use_default_kernel_priors: If True, use BoTorch's default priors and constraints for kernel
            hyperparameters (lengthscale, outputscale) and noise. This enables Maximum A Posteriori (MAP)
            estimation with dimension-scaled priors from [Hvarfner2024vanilla], matching SingleTaskGP's
            default behavior. If False (default), uses Maximum Likelihood Estimation (MLE) without priors.

            **Interaction with freeze_xxx flags**: When `use_default_kernel_priors=True`, priors are
            applied to all hyperparameters. You can then selectively freeze specific parameters using
            `freeze_lengthscale`, `freeze_outputscale`, or `freeze_noise` in `hyperparameters`. Only
            the non-frozen parameters will be trained via MAP; frozen parameters retain their initial
            values (either defaults or values you specify in `hyperparameters`).

            Example: `use_default_kernel_priors=True` with `freeze_noise=True` will train lengthscale
            and outputscale via MAP while keeping noise fixed.

            Recommended: True for learned hyperparameters, False for GP prior benchmarks or pure MLE.
            Note: When True with use_outcome_transform=True, no ScaleKernel is used (matches SingleTaskGP).
            When True with use_outcome_transform=False, ScaleKernel is added for fixed hyperparameter support.

    Returns:
        A fitted (or unfitted if fit_model=False) SingleTaskGP model.

    Example:
        >>> # Standard model with learned hyperparameters
        >>> train_X = torch.rand(10, 2)
        >>> train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
        >>> model = build_and_fit_gp(train_X, train_Y, kernel="rbf")

        >>> # Model with fixed near-zero noise (for Matheron path interpolation)
        >>> model = build_and_fit_gp(
        ...     train_X, train_Y,
        ...     kernel="rbf",
        ...     hyperparameters={"noise": 1e-6, "freeze_noise": True}
        ... )

        >>> # Model with fixed lengthscale and outputscale
        >>> model = build_and_fit_gp(
        ...     train_X, train_Y,
        ...     kernel="matern52",
        ...     hyperparameters={
        ...         "lengthscale": 0.2,
        ...         "outputscale": 1.0,
        ...         "noise": 1e-4,
        ...         "freeze_lengthscale": True,
        ...         "freeze_outputscale": True,
        ...         "freeze_noise": True,
        ...     }
        ... )
    """
    # Validate inputs
    if train_X.dim() != 2:
        raise ValueError(f"train_X must be 2-dimensional, got shape {train_X.shape}")

    # Ensure train_Y is 2-dimensional (n, 1)
    if train_Y.dim() == 1:
        train_Y = train_Y.unsqueeze(-1)
    elif train_Y.dim() == 2 and train_Y.shape[-1] != 1:
        raise ValueError(
            f"train_Y must have shape (n, 1) or (n,), got shape {train_Y.shape}"
        )

    # Validate input_bounds
    if input_bounds is None:
        raise ValueError(
            "input_bounds must be provided. This is required for benchmarking to ensure "
            "consistent input normalization. Pass a (2, d) tensor with lower and upper bounds."
        )
    if input_bounds.dim() != 2 or input_bounds.shape[0] != 2:
        raise ValueError(
            f"input_bounds must have shape (2, d), got shape {input_bounds.shape}"
        )
    if input_bounds.shape[1] != train_X.shape[1]:
        raise ValueError(
            f"input_bounds dimension {input_bounds.shape[1]} does not match "
            f"train_X dimension {train_X.shape[1]}"
        )

    # Validate transform settings (use_outcome_transform must be explicitly set)
    if use_outcome_transform is None:
        raise ValueError(
            "use_outcome_transform must be explicitly set to True or False. "
            "Recommended: True for learned hyperparameters, False for GP prior benchmarks."
        )

    # Prepare transforms
    input_transform = None
    outcome_transform = None

    if use_input_transform:
        from botorch.models.transforms import Normalize
        # Use explicit bounds to ensure consistent normalization across iterations
        # This prevents the transform from learning bounds from training data
        input_transform = Normalize(d=train_X.shape[-1], bounds=input_bounds)

    if use_outcome_transform:
        from botorch.models.transforms import Standardize
        outcome_transform = Standardize(m=train_Y.shape[-1])

    # Create covar_module BEFORE SingleTaskGP initialization
    dim = train_X.shape[-1]

    if use_default_kernel_priors:
        # Use BoTorch's default priors (matches SingleTaskGP behavior)
        from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
        from gpytorch.priors import GammaPrior

        # Validate kernel specification
        if kernel.lower() not in ["rbf", "matern", "matern52"]:
            raise ValueError(
                f"When use_default_kernel_priors=True, only 'rbf' and 'matern52' kernels are supported. "
                f"Got: {kernel}. For other kernels (including 'tanimoto'), use use_default_kernel_priors=False."
            )

        if use_outcome_transform:
            # Match SingleTaskGP exactly: base kernel only (no ScaleKernel)
            # Standardize handles output scaling, so no outputscale parameter needed
            covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=dim,
                use_rbf_kernel=(kernel.lower() == "rbf")
            )
        else:
            # For GP priors: need ScaleKernel to support fixed outputscale
            base_kernel = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=dim,
                use_rbf_kernel=(kernel.lower() == "rbf")
            )
            # Add ScaleKernel with outputscale prior
            covar_module = ScaleKernel(
                base_kernel,
                outputscale_prior=GammaPrior(2.0, 0.15)
            )
    else:
        # No priors: create kernel manually (current behavior)
        if kernel.lower() == "rbf":
            base_kernel = RBFKernel(ard_num_dims=dim)
        elif kernel.lower() == "matern" or kernel.lower() == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=dim)
        elif kernel.lower() == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=dim)
        elif kernel.lower() == "matern12":
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=dim)
        elif kernel.lower() == "tanimoto":
            try:
                from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
            except ImportError:
                raise ImportError(
                    "Tanimoto kernel requires gauche. Install with: pip install gauche"
                )
            base_kernel = TanimotoKernel()
        else:
            raise ValueError(
                f"Unsupported kernel: {kernel}. Supported options: "
                "'rbf', 'matern', 'matern52', 'matern32', 'matern12', 'tanimoto'"
            )

        # Always wrap with ScaleKernel to support outputscale parameter
        covar_module = ScaleKernel(base_kernel)

    # Create the GP model with custom covar_module
    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    # Set hyperparameters if provided
    if hyperparameters is not None:
        # Handle lengthscale
        if "lengthscale" in hyperparameters:
            lengthscale = hyperparameters["lengthscale"]
            if isinstance(lengthscale, (int, float, list)):
                lengthscale = torch.tensor(lengthscale, dtype=train_X.dtype, device=train_X.device)

            # Access base kernel (handle ScaleKernel wrapper)
            if hasattr(model.covar_module, "base_kernel"):
                base_kernel = model.covar_module.base_kernel
            else:
                base_kernel = model.covar_module

            base_kernel.lengthscale = lengthscale

            # Optionally freeze
            if hyperparameters.get("freeze_lengthscale", False):
                base_kernel.raw_lengthscale.requires_grad = False

        # Handle outputscale (only for ScaleKernel)
        if "outputscale" in hyperparameters:
            if hasattr(model.covar_module, "outputscale"):
                outputscale = hyperparameters["outputscale"]
                if isinstance(outputscale, (int, float)):
                    outputscale = torch.tensor(outputscale, dtype=train_X.dtype, device=train_X.device)
                model.covar_module.outputscale = outputscale

                # Optionally freeze
                if hyperparameters.get("freeze_outputscale", False):
                    model.covar_module.raw_outputscale.requires_grad = False
            else:
                raise ValueError(
                    "Cannot set outputscale: model kernel does not have outputscale "
                    "(not wrapped with ScaleKernel)"
                )

        # Handle noise
        if "noise" in hyperparameters:
            noise = hyperparameters["noise"]
            if isinstance(noise, (int, float)):
                noise = torch.tensor(noise, dtype=train_X.dtype, device=train_X.device)

            # Optionally freeze and remove prior
            if hyperparameters.get("freeze_noise", False):
                # Remove noise prior first (frozen parameters shouldn't have priors)
                if hasattr(model.likelihood.noise_covar, 'noise_prior'):
                    delattr(model.likelihood.noise_covar, 'noise_prior')
                # Set the noise value and freeze it
                model.likelihood.noise = noise
                model.likelihood.noise_covar.raw_noise.requires_grad = False
            else:
                # Learnable noise: just set the value
                model.likelihood.noise = noise

        # Handle mean constant (for exact replay)
        if "mean_constant" in hyperparameters:
            mean_constant = hyperparameters["mean_constant"]
            if isinstance(mean_constant, (int, float)):
                mean_constant = torch.tensor([[mean_constant]], dtype=train_X.dtype, device=train_X.device)
            if hasattr(model.mean_module, "constant"):
                model.mean_module.constant.data = mean_constant
            # If mean_constant is provided, freeze it by default (for exact replay)
            if hyperparameters.get("freeze_mean", True):
                for param in model.mean_module.parameters():
                    param.requires_grad = False

        # If all main hyperparameters are frozen, also freeze the mean function
        # to ensure no parameters are optimized
        # Check: freeze_lengthscale AND freeze_noise AND (freeze_outputscale OR no outputscale)
        has_outputscale = hasattr(model.covar_module, "outputscale")
        outputscale_frozen = hyperparameters.get("freeze_outputscale", False) or not has_outputscale
        if (hyperparameters.get("freeze_lengthscale", False) and
            outputscale_frozen and
            hyperparameters.get("freeze_noise", False)):
            for param in model.mean_module.parameters():
                param.requires_grad = False

    # Fit the model if requested (but skip if all parameters are frozen)
    if fit_model:
        # Check if any parameters require gradients
        trainable_params = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]
        has_trainable_params = len(trainable_params) > 0

        if verbose and not has_trainable_params:
            print("  All parameters frozen - skipping model fitting")

        if verbose and has_trainable_params:
            print(f"  Trainable parameters: {trainable_params}")

        if has_trainable_params:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, max_retries=max_retries)

    # Print hyperparameters if verbose
    if verbose:
        print("Model hyperparameters:")

        # Get lengthscale (if kernel has one - Tanimoto doesn't)
        if hasattr(model.covar_module, "base_kernel"):
            base_kernel = model.covar_module.base_kernel
        else:
            base_kernel = model.covar_module

        if hasattr(base_kernel, "lengthscale") and hasattr(base_kernel, "raw_lengthscale"):
            lengthscale = base_kernel.lengthscale
            frozen_ls = not base_kernel.raw_lengthscale.requires_grad
            freeze_str = " (frozen)" if frozen_ls else ""
            if lengthscale.numel() == 1:
                print(f"  Lengthscale: {lengthscale.item():.6f}{freeze_str}")
            else:
                print(f"  Lengthscale (ARD): {lengthscale.squeeze().tolist()}{freeze_str}")
        else:
            print(f"  Lengthscale: N/A (kernel type: {base_kernel.__class__.__name__})")

        # Get outputscale
        if hasattr(model.covar_module, "outputscale"):
            outputscale = model.covar_module.outputscale.item()
            frozen_os = not model.covar_module.raw_outputscale.requires_grad
            freeze_str = " (frozen)" if frozen_os else ""
            print(f"  Outputscale: {outputscale:.6f}{freeze_str}")

        # Get noise
        noise = model.likelihood.noise.item()
        frozen_noise = not model.likelihood.noise_covar.raw_noise.requires_grad
        freeze_str = " (frozen)" if frozen_noise else ""
        print(f"  Noise: {noise:.6e}{freeze_str}")

    return model


def get_model_hyperparameters(model: SingleTaskGP) -> Dict[str, Any]:
    r"""Extract hyperparameters from a fitted SingleTaskGP model.

    This utility function extracts the current hyperparameter values from a
    fitted model, which can be useful for:
    - Logging/tracking hyperparameters during experiments
    - Debugging model behavior
    - Creating fixed hyperparameter configurations for reproducibility

    Args:
        model: A SingleTaskGP model (fitted or unfitted).

    Returns:
        A dictionary containing the model's hyperparameters:
            - "lengthscale": Tensor of shape `(d,)` for ARD or scalar
            - "outputscale": float (if model uses ScaleKernel)
            - "noise": float
            - "kernel_type": str indicating the kernel type

    Example:
        >>> model = build_and_fit_gp(train_X, train_Y)
        >>> hyperparams = get_model_hyperparameters(model)
        >>> print(hyperparams)
        {'lengthscale': tensor([0.2345]), 'outputscale': 1.234, 'noise': 0.001, ...}
    """
    hyperparams = {}

    # Get lengthscale (handle ScaleKernel wrapper)
    if hasattr(model.covar_module, "base_kernel"):
        base_kernel = model.covar_module.base_kernel
        kernel_name = base_kernel.__class__.__name__
    else:
        base_kernel = model.covar_module
        kernel_name = base_kernel.__class__.__name__

    if hasattr(base_kernel, "lengthscale") and hasattr(base_kernel, "raw_lengthscale"):
        lengthscale = base_kernel.lengthscale.detach().clone()
        hyperparams["lengthscale"] = lengthscale.squeeze() if lengthscale.numel() > 1 else lengthscale.item()
        hyperparams["lengthscale_frozen"] = not base_kernel.raw_lengthscale.requires_grad

    # Get outputscale (if ScaleKernel)
    if hasattr(model.covar_module, "outputscale"):
        hyperparams["outputscale"] = model.covar_module.outputscale.item()
        hyperparams["outputscale_frozen"] = not model.covar_module.raw_outputscale.requires_grad

    # Get noise
    hyperparams["noise"] = model.likelihood.noise.item()
    hyperparams["noise_frozen"] = not model.likelihood.noise_covar.raw_noise.requires_grad

    # Get mean constant (important for replay)
    if hasattr(model.mean_module, "constant"):
        hyperparams["mean_constant"] = model.mean_module.constant.detach().clone().squeeze().item()

    # Add kernel type
    hyperparams["kernel_type"] = kernel_name

    return hyperparams
