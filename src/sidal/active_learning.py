import torch
import gpytorch
from functools import partial
from torch import Tensor
from typing import Optional, Callable, NamedTuple

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from botorch.utils.transforms import (
    t_batch_mode_transform,
    average_over_ensemble_models,
)

from .design_space import DesignSpace


class FantasyVarianceCache(NamedTuple):
    """Cached kernel computations for fantasy variance calculations.

    These matrices are computed once and reused for all candidate evaluations.
    """
    L_train: Tensor  # Cholesky of K(train, train) + noise
    K_star_train: Tensor  # K(mc_points, train)
    Ainv_Kst: Tensor  # L_train^{-1} @ K_star_train.T
    prior_var_diag: Tensor  # diag(K(mc_points, mc_points))
    red_train_diag: Tensor  # Variance reduction from training data
    var_before: Tensor  # Current posterior variance at mc_points


class FantasyVarianceMixin:
    """Mixin class providing shared functionality for active learning acquisition functions.

    This mixin provides:
    1. Integration points setup from DesignSpace
    2. Efficient fantasy posterior variance computation using direct GP kernel formulas

    The key insight is that for standard GPs, the posterior variance after
    observing a new point x depends only on x's location, not on the observed
    value y. This allows efficient batch computation without actually
    conditioning on fantasy observations.

    Classes using this mixin should:
    1. Have a `model` attribute (SingleTaskGP or compatible)
    2. Call `_setup_reference_points(space)` in __init__ to set up integration points
    3. Call `_prepare_fantasy_variance_cache()` once per forward pass
    4. Call `_compute_fantasy_variance()` for batched candidate evaluation

    The mixin handles:
    - Integration points: Extracted from DesignSpace for consistency across algorithms
    - Input transform: Transforms X and reference_points to match train_X coordinate system
    - Output transform: Scales variance from standardized to original Y space
    - Efficient Schur complement computation for fantasy variance
    """

    def _setup_reference_points(self, space: "DesignSpace") -> None:
        """Set up integration points from a DesignSpace.

        This method should be called in __init__ of classes using this mixin.
        It ensures all acquisition functions use the same integration points
        from the DesignSpace for fair comparison.

        Args:
            space: DesignSpace object (discrete or continuous)
        """
        self.space = space
        if space.is_discrete:
            self.reference_points = space.candidates
        else:
            # For continuous spaces, generate reference points if not already set
            # For simple generators (e.g., Sobol), this can be done at init time
            if space._reference_points is None and space._reference_point_generator is not None:
                # Generate reference points (pass None for energy_distribution,
                # simple generators like Sobol will ignore it via **kwargs)
                space._reference_points = space._reference_point_generator(
                    bounds=space.bounds,
                    **space._reference_point_generator_kwargs,
                )
            self.reference_points = space.reference_points

    def _get_reference_points(self) -> Tensor:
        """Get integration/target points for variance computation."""
        if hasattr(self, 'reference_points'):
            return self.reference_points
        else:
            raise AttributeError(
                "FantasyVarianceMixin requires 'reference_points' attribute. "
                "Call _setup_reference_points(space) in __init__."
            )

    @property
    def integration_grid(self) -> Tensor:
        """Unified interface for accessing integration points."""
        return self._get_reference_points()

    def _prepare_fantasy_variance_cache(
        self,
        apply_output_transform: bool = True,
    ) -> FantasyVarianceCache:
        """Prepare cached kernel computations for fantasy variance.

        This method computes and caches matrices that are shared across all
        candidate evaluations, including:
        - Cholesky decomposition of training kernel
        - Current posterior variance at integration points

        Args:
            apply_output_transform: If True, scale variance to original Y space.
                Set to False if you need raw (standardized) variance.

        Returns:
            FantasyVarianceCache containing precomputed matrices.
        """
        model = self.model
        mc_points = self._get_reference_points()

        # Get training data (already in transformed space if input_transform exists)
        train_X = model.train_inputs[0]
        model_dtype = train_X.dtype
        model_device = train_X.device
        n_train = train_X.shape[0]

        # Apply input transform if present
        if hasattr(model, 'input_transform') and model.input_transform is not None:
            mc_points_transformed = model.input_transform(mc_points).to(
                dtype=model_dtype, device=model_device
            )
        else:
            mc_points_transformed = mc_points.to(dtype=model_dtype, device=model_device)

        # Get covariance module and noise
        covar_module = model.covar_module
        noise_var = model.likelihood.noise.item()

        # Compute kernel matrices
        K_star_train = covar_module(mc_points_transformed, train_X).to_dense()
        K_train_train = covar_module(train_X, train_X).to_dense()
        K_train_train_noisy = K_train_train + noise_var * torch.eye(
            n_train, dtype=model_dtype, device=model_device
        )
        prior_var_diag = covar_module(
            mc_points_transformed, mc_points_transformed
        ).diagonal()

        # Cholesky decomposition (reused for all candidates)
        L_train = torch.linalg.cholesky(K_train_train_noisy)

        # Current posterior variance reduction from training data
        Ainv_Kst = torch.cholesky_solve(K_star_train.t(), L_train)  # (n_train, n_mc)
        red_train_diag = (K_star_train * Ainv_Kst.t()).sum(dim=1)  # (n_mc,)

        # Current posterior variance
        var_before = prior_var_diag - red_train_diag  # (n_mc,)

        # Apply output transform if present
        if apply_output_transform:
            if hasattr(model, 'outcome_transform') and model.outcome_transform is not None:
                if hasattr(model.outcome_transform, 'stdvs'):
                    scale = model.outcome_transform.stdvs ** 2
                    var_before = var_before * scale

        return FantasyVarianceCache(
            L_train=L_train,
            K_star_train=K_star_train,
            Ainv_Kst=Ainv_Kst,
            prior_var_diag=prior_var_diag,
            red_train_diag=red_train_diag,
            var_before=var_before,
        )

    def _compute_fantasy_variance(
        self,
        X: Tensor,
        cache: FantasyVarianceCache,
        apply_output_transform: bool = True,
    ) -> Tensor:
        """Compute fantasy variance for candidates (vectorized).

        Uses Schur complement for efficient computation:
        Var[f(x*) | D ∪ {x}] = Var[f(x*) | D] - k(x*, x)² / (k(x,x) + σ² - k(x,X)A⁻¹k(X,x))

        Args:
            X: (n_candidates, d) candidate points
            cache: Precomputed kernel matrices from _prepare_fantasy_variance_cache()
            apply_output_transform: If True, scale variance to original Y space.

        Returns:
            var_after: (n_candidates, n_mc) fantasy variances for each candidate
        """
        model = self.model
        mc_points = self._get_reference_points()

        # Get training data
        train_X = model.train_inputs[0]
        model_dtype = train_X.dtype
        model_device = train_X.device

        # Apply input transform if present
        if hasattr(model, 'input_transform') and model.input_transform is not None:
            X_transformed = model.input_transform(X).to(
                dtype=model_dtype, device=model_device
            )
            mc_points_transformed = model.input_transform(mc_points).to(
                dtype=model_dtype, device=model_device
            )
        else:
            X_transformed = X.to(dtype=model_dtype, device=model_device)
            mc_points_transformed = mc_points.to(dtype=model_dtype, device=model_device)

        # Get covariance module and noise
        covar_module = model.covar_module
        noise_var = model.likelihood.noise.item()

        # Compute kernel matrices for all candidates at once
        K_star_X = covar_module(mc_points_transformed, X_transformed).to_dense()  # (n_mc, n_cand)
        K_X_train = covar_module(X_transformed, train_X).to_dense()  # (n_cand, n_train)
        K_X_X_diag = covar_module(X_transformed, X_transformed).diagonal()  # (n_cand,)

        # Schur complement for all candidates
        Ainv_Bt = torch.cholesky_solve(K_X_train.t(), cache.L_train)  # (n_train, n_cand)
        S = K_X_X_diag + noise_var - (K_X_train * Ainv_Bt.t()).sum(dim=1)  # (n_cand,)

        # Additional variance reduction
        K_star_train_Ainv_Bt = cache.K_star_train @ Ainv_Bt  # (n_mc, n_cand)
        numerator = (K_star_X - K_star_train_Ainv_Bt) ** 2  # (n_mc, n_cand)
        additional_red = numerator / S.unsqueeze(0)  # (n_mc, n_cand)

        # Fantasy variance for each candidate: (n_mc, n_cand) -> transpose to (n_cand, n_mc)
        var_after = (
            cache.prior_var_diag.unsqueeze(1)
            - cache.red_train_diag.unsqueeze(1)
            - additional_red
        ).t()  # (n_cand, n_mc)

        # Apply output transform if present
        if apply_output_transform:
            if hasattr(model, 'outcome_transform') and model.outcome_transform is not None:
                if hasattr(model.outcome_transform, 'stdvs'):
                    scale = model.outcome_transform.stdvs ** 2
                    var_after = var_after * scale

        return var_after


class qExpectedPredictiveInformationGain(FantasyVarianceMixin, AcquisitionFunction):
    r"""Expected Predictive Information Gain (EPIG) for Active Learning.

    This acquisition function quantifies the expected information gain about
    predictions at target points (rather than model parameters). It measures
    how much observing a candidate point x reduces uncertainty about predictions
    at a set of target points x_*.

    For Gaussian processes, EPIG has the closed form:
        EPIG(x) = E_{x_*}[0.5 * log(σ²(x_* | D) / σ²(x_* | D ∪ x))]

    where the expectation is over the target distribution (typically uniform over
    the search space, represented by reference_points from the DesignSpace).

    Compared to BALD (mutual information with parameters), EPIG is prediction-oriented
    and naturally accounts for the input distribution, avoiding prioritizing outliers
    that reduce parameter uncertainty without improving predictive performance.

    See "Prediction-Oriented Bayesian Active Learning" (Bickford Smith et al., AISTATS 2023).
    """

    def __init__(
        self,
        model: Model,
        space: "DesignSpace",
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""Expected Predictive Information Gain.

        Args:
            model: A fitted model.
            space: DesignSpace object defining the integration points.
                For discrete spaces, uses space.candidates as integration points.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        self._setup_reference_points(space)
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        """Compute Expected Predictive Information Gain.

        EPIG(x) = E_{x*}[0.5 * log(σ²_before(x*) / σ²_after(x*))]

        where σ²_before is current posterior variance and σ²_after is fantasy
        posterior variance after observing at x.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of candidate points.

        Returns:
            A `batch_shape`-dim Tensor of EPIG values.
        """
        # X shape: batch_shape x q x d
        batch_shape = X.shape[:-2]
        q = X.shape[-2]

        if q != 1:
            raise NotImplementedError("EPIG only supports q=1 for now.")

        # Prepare cache with shared kernel computations
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=True)
        var_before = cache.var_before  # (n_mc,)

        # Flatten batch dimensions: (batch_shape, 1, d) -> (n_batch, d)
        n_batch = X[..., 0, 0].numel() if batch_shape else 1
        X_flat = X.reshape(n_batch, -1)

        # Vectorized computation for all candidates
        var_after_all = self._compute_fantasy_variance(
            X_flat, cache, apply_output_transform=True
        )  # (n_batch, n_mc)

        # Compute information gain: 0.5 * log(var_before / var_after)
        eps = 1e-10
        log_ratio = torch.log((var_before + eps) / (var_after_all + eps))
        info_gain = 0.5 * log_ratio  # (n_batch, n_mc)

        # Average over integration points
        result_flat = info_gain.mean(dim=-1)  # (n_batch,)

        # Reshape back to batch_shape
        if batch_shape:
            return result_flat.reshape(batch_shape)
        else:
            return result_flat.squeeze(0)


class qNegIntegratedPosteriorVariance(FantasyVarianceMixin, AcquisitionFunction):
    r"""Batch Integrated Negative Posterior Variance for Active Learning.

    This acquisition function quantifies the (negative) integrated posterior variance
    (excluding observation noise, computed using MC integration) of the model.
    In that, it is a proxy for global model uncertainty, and thus purely focused on
    "exploration", rather the "exploitation" of many of the classic Bayesian
    Optimization acquisition functions.

    See [Seo2014activedata]_, [Chen2014seqexpdesign]_, and [Binois2017repexp]_.
    """

    def __init__(
        self,
        model: Model,
        space: "DesignSpace",
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        r"""q-Integrated Negative Posterior Variance.

        Args:
            model: A fitted model.
            space: DesignSpace object defining the integration points.
                For discrete spaces, uses space.candidates as integration points.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)
        self._setup_reference_points(space)
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNIPV.

        IMSE(x) = -E_{x*}[σ²(x* | D ∪ {x})]

        Returns negative mean fantasy variance at integration points.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of candidate points.

        Returns:
            A `batch_shape`-dim Tensor of Integrated Negative Posterior Variance values.
        """
        # X shape: batch_shape x q x d
        batch_shape = X.shape[:-2]
        q = X.shape[-2]

        if q != 1:
            raise NotImplementedError("IMSE only supports q=1 for now.")

        # Prepare cache with shared kernel computations
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=True)

        # Flatten batch dimensions: (batch_shape, 1, d) -> (n_batch, d)
        n_batch = X[..., 0, 0].numel() if batch_shape else 1
        X_flat = X.reshape(n_batch, -1)

        # Vectorized computation for all candidates
        var_after_all = self._compute_fantasy_variance(
            X_flat, cache, apply_output_transform=True
        )  # (n_batch, n_mc)

        # Negative mean variance over integration points
        result_flat = -var_after_all.mean(dim=-1)  # (n_batch,)

        # Reshape back to batch_shape
        if batch_shape:
            return result_flat.reshape(batch_shape)
        else:
            return result_flat.squeeze(0)


class PosteriorVariance(AcquisitionFunction):
    r"""Posterior Variance for Active Learning.

    This acquisition function returns the posterior variance (excluding
    observation noise) at the candidate points. It selects points where
    the model is most uncertain.

    This is a simple, local measure of uncertainty (compared to the
    integrated version qNegIntegratedPosteriorVariance).
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Posterior Variance acquisition function.

        Args:
            model: A fitted model.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, return positive variance (for maximization).
                If False, return negative variance (matches convention of other acqfs).
        """
        super().__init__(model=model)
        self.posterior_transform = posterior_transform
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate PosteriorVariance on the candidate set `X`.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim Tensor of t-batches with `1` `d`-dim
                design point each.

        Returns:
            A `(b1 x ... bk)`-dim Tensor of posterior variance values at the
            given design points `X`.
        """
        posterior = self.model.posterior(
            X, posterior_transform=self.posterior_transform
        )
        variance = posterior.variance.view(X.shape[:-2])
        return variance if self.maximize else -variance
    

class VarianceConstraintMixin:
    """
    Mixin providing variance-based constraint set functionality.

    The constraint set X̄_t restricts candidates to high-uncertainty regions,
    ensuring sufficient exploration during active learning.

    Supported constraint types:
        - "energy_weighted_variance": X̄_t := {x ∈ X | σ²(x) ≥ E_P[σ²]}

    Classes using this mixin must have:
        - self.model: GP model
        - self.reference_points: (n, d) tensor
        - self._energy_dist: BoltzmannEnergyDistribution (for weighted variance)
    """

    VALID_CONSTRAINT_TYPES = (None, "energy_weighted_variance")

    def _init_constraint(
        self,
        constraint_type: Optional[str],
    ) -> None:
        """
        Initialize constraint set parameters.

        Args:
            constraint_type: Type of constraint set, or None to disable:
                - None: No constraint, all points valid
                - "energy_weighted_variance": σ²(x) ≥ E_P[σ²]
        """
        if constraint_type is not None and constraint_type not in self.VALID_CONSTRAINT_TYPES:
            raise ValueError(
                f"constraint_type must be one of {self.VALID_CONSTRAINT_TYPES}, "
                f"got '{constraint_type}'"
            )
        self._constraint_type = constraint_type
        self._cached_constraint_mask = None
        self._cached_constraint_train_inputs = None
        self._cached_variance_threshold = None  # Cache threshold for continuous constraint checking

    def _constraint_needs_recompute(self) -> bool:
        """Check if constraint set needs recomputation based on model changes."""
        current_train_inputs = self.model.train_inputs[0]
        if self._cached_constraint_train_inputs is None:
            return True
        if current_train_inputs.shape != self._cached_constraint_train_inputs.shape:
            return True
        if not torch.allclose(current_train_inputs, self._cached_constraint_train_inputs):
            return True
        return False

    def _compute_constraint_mask(
        self,
        sigma_sq: Tensor,
        weight_distribution: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute constraint set mask at integration points.

        Args:
            sigma_sq: (n_reference_points,) posterior variances
            weight_distribution: (n_reference_points,) optional weights for
                "energy_weighted_variance" type

        Returns:
            mask: (n_reference_points,) boolean mask for X̄_t
        """
        if self._constraint_type == "energy_weighted_variance":
            if weight_distribution is None:
                threshold = sigma_sq.mean()
            else:
                threshold = (weight_distribution * sigma_sq).sum()
            mask = sigma_sq >= threshold

        else:
            raise ValueError(f"Unknown constraint_type: {self._constraint_type}")

        # Cache threshold for continuous constraint checking
        self._cached_variance_threshold = threshold

        # Check for empty feasible set
        if not mask.any():
            raise RuntimeError(
                f"VarianceConstraintMixin has no feasible points. "
                f"constraint_type='{self._constraint_type}', "
                f"sigma_sq range=[{sigma_sq.min():.4e}, {sigma_sq.max():.4e}]"
            )

        return mask

    @property
    def constraint_set_mask_at_reference_points(self) -> Tensor:
        """
        Get constraint set X̄_t as a boolean mask at integration points.

        Returns all True if constraint_type is None.
        """
        if self._constraint_type is None:
            return torch.ones(
                len(self.reference_points),
                dtype=torch.bool,
                device=self.reference_points.device
            )

        if self._constraint_needs_recompute():
            _, sigma = self._energy_dist.compute_posterior_stats_at_reference_points()
            sigma_sq = sigma ** 2

            weight_dist = None
            if self._constraint_type == "energy_weighted_variance":
                weight_dist = self._energy_dist.compute_integration_weights()

            self._cached_constraint_mask = self._compute_constraint_mask(
                sigma_sq, weight_distribution=weight_dist
            )
            self._cached_constraint_train_inputs = self.model.train_inputs[0].clone()
            self._energy_dist.invalidate_cache()

        return self._cached_constraint_mask

    @property
    def variance_threshold(self) -> Optional[Tensor]:
        """
        Get the variance threshold for the constraint set.

        Returns None if constraint_type is None.
        Triggers constraint computation if needed.

        Returns:
            threshold: scalar tensor, or None if no constraint
        """
        if self._constraint_type is None:
            return None
        # Trigger computation if needed
        _ = self.constraint_set_mask_at_reference_points
        return self._cached_variance_threshold

    def get_variance_constraint_callable(self) -> Optional[callable]:
        """
        Get a callable constraint function for use with continuous optimization.

        The constraint is: σ²(x) - threshold ≥ 0 (feasible region = high variance)

        Returns:
            callable: Function f(X) -> Tensor where f(X) ≥ 0 means feasible,
                     or None if no constraint is active.

        Usage with optimize_acqf:
            constraint_fn = acq.get_variance_constraint_callable()
            if constraint_fn is not None:
                new_x, acq_val = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=512,
                    nonlinear_inequality_constraints=[(constraint_fn, True)],
                )
        """
        if self._constraint_type is None:
            return None

        # Ensure threshold is computed
        threshold = self.variance_threshold
        model = self.model

        def variance_constraint(X: Tensor) -> Tensor:
            """
            Constraint function: σ²(x) - threshold ≥ 0

            Args:
                X: (d,) single point (is_intra_point=True)

            Returns:
                scalar constraint value where ≥ 0 means feasible
            """
            # is_intra_point=True means X is always (d,)
            posterior = model.posterior(X.unsqueeze(0))  # (1, d) -> posterior
            var = posterior.variance.squeeze()  # scalar
            return var - threshold

        return variance_constraint

    def _apply_constraint_to_acquisition(
        self,
        X: Tensor,
        acq_values: Tensor,
    ) -> Tensor:
        """
        Apply constraint set X̄_t to acquisition values.

        Sets acquisition to -inf for points not in X̄_t.

        The constraint threshold is computed on reference points (via variance_threshold
        property) and applied to X by computing σ²(X) directly. This is O(|X|)
        instead of O(|X| * |ref| * d) pairwise matching.

        Args:
            X: (batch, d) candidate points
            acq_values: (batch,) acquisition values

        Returns:
            acq: (batch,) acquisition values with -inf for points outside X̄_t
        """
        # Get variance threshold (computed on reference points, cached)
        threshold = self.variance_threshold
        if threshold is None:
            # No constraint
            return acq_values

        # Compute variance at X directly - O(|X|) instead of O(|X| * |ref| * d)
        with torch.no_grad():
            posterior = self.model.posterior(X)
            sigma_sq_X = posterior.variance.squeeze(-1)  # (batch,)

        # Check constraint: σ²(x) >= threshold
        in_constraint = sigma_sq_X >= threshold

        # Set acquisition to -inf for points not in constraint set
        acq = acq_values.clone()
        acq[~in_constraint] = float('-inf')

        return acq


class BoltzmannEnergyDistribution:
    """
    Computes Boltzmann-like energy distributions over a design space.

    The distribution has the form:
        P̃(x) ∝ exp(λ · g(x) / T + b(x))

    where:
        - g(x) is the energy function
        - λ is boltzmann_sign (+1 for maximization, -1 for minimization)
        - T is temperature
        - b(x) is an optional bias function

    **Approximation Modes:**

    1. `base_approximation` (deterministic/analytic):
        - None: No unknown part in energy term, just exp(b(x))  (IMSE-like)
        - "mean": Use posterior mean μ(x)
        - "zeroth_order_analytic": E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)

    2. `use_thompson_sampling` (stochastic):
        - True with base_approximation=None: Pure Thompson sampling exp(λf/T + b)
          where f ~ P(f|D) is a sample from the posterior

    Note: Thompson sampling and base_approximation are mutually exclusive.
    """

    VALID_BASE_APPROXIMATIONS = (None, "mean", "zeroth_order_analytic")

    def __init__(
        self,
        model: SingleTaskGP,
        reference_points: Tensor,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        base_approximation: Optional[str] = "zeroth_order_analytic",
        use_thompson_sampling: bool = False,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        seed: Optional[int] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
        thompson_path: Optional[Callable[[Tensor], Tensor]] = None,
        sample_approach: str = "ff",
    ):
        """
        Args:
            model: Fitted GP model
            reference_points: (n, d) points where distribution is evaluated
            temperature: Temperature parameter T (default: 1.0)
            boltzmann_sign: Sign λ in exp(λ·g/T) (default: -1.0 for minimization)
            base_approximation: Deterministic/analytic base approximation:
                - None: No energy term, just exp(b(x))
                - "mean": posterior mean μ(x)
                - "zeroth_order_analytic": E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)
                  (default)
            use_thompson_sampling: If True, use pure Thompson sampling: sample
                f ~ P(f|D) and compute exp(λf/T + b). Requires base_approximation=None.
            bias_fn: Bias function b(x). If None, uses zero bias.
            seed: Random seed for Thompson sampling (ignored if thompson_path provided)
            stabilize_log_density: If True, stabilize log density to prevent
                exponential overflow. For zeroth_order_analytic, subtracts
                (λ/T)²·σ²_max/2. For Thompson sampling, subtracts max(log_density).
                This preserves argmin/argmax ranking. Recommended when posterior
                variance is large or function values have extreme range.
                Default: True. Only supported with zeroth_order_analytic or
                use_thompson_sampling=True.
            num_rff_features: Number of RFF features for Thompson sampling pathwise
                approximation. Default: 4096. Ignored if thompson_path provided.
            thompson_path: Optional external Thompson sample path. If provided,
                use this instead of creating a new one. This is useful for
                TS-SID-iVAR where SMC and acquisition function should use the
                same Thompson sample. The callable should map (n, d) -> (1, n, 1).
            sample_approach: Approach for Thompson sampling:
                - "ff": Fourier features / pathwise sampling (default). Works with
                  stationary kernels (RBF, Matern). More efficient for continuous domains.
                - "dmvn": Direct multivariate normal sampling. Works with any kernel
                  including non-stationary ones (Tanimoto). Better for pool-based AL
                  with finite candidates.
        """
        if base_approximation is not None and base_approximation not in self.VALID_BASE_APPROXIMATIONS:
            raise ValueError(
                f"base_approximation must be one of {self.VALID_BASE_APPROXIMATIONS}, "
                f"got '{base_approximation}'"
            )

        if sample_approach not in ("ff", "dmvn"):
            raise ValueError(
                f"sample_approach must be 'ff' or 'dmvn', got '{sample_approach}'"
            )

        if use_thompson_sampling and base_approximation is not None:
            raise ValueError(
                "Thompson sampling cannot be combined with base_approximation. "
                "Use base_approximation=None for pure Thompson sampling, or "
                "use_thompson_sampling=False for deterministic approximations."
            )

        # stabilize_log_density is only meaningful for zeroth_order_analytic and Thompson
        # For other modes, the log density doesn't have terms that can overflow
        if stabilize_log_density and not (
            base_approximation == "zeroth_order_analytic" or use_thompson_sampling
        ):
            raise ValueError(
                f"stabilize_log_density=True is only supported with "
                f"base_approximation='zeroth_order_analytic' or use_thompson_sampling=True. "
                f"Got base_approximation='{base_approximation}', use_thompson_sampling={use_thompson_sampling}."
            )

        self.model = model
        self.reference_points = reference_points
        self.temperature = temperature
        self.boltzmann_sign = boltzmann_sign
        self.base_approximation = base_approximation
        self.use_thompson_sampling = use_thompson_sampling
        self.seed = seed
        self.stabilize_log_density = stabilize_log_density
        self.num_rff_features = num_rff_features
        self.sample_approach = sample_approach

        # Set up bias function
        if bias_fn is None:
            self.bias_fn = lambda X: torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        else:
            self.bias_fn = bias_fn

        # External Thompson path (for sharing between SMC and acquisition function)
        self._external_thompson_path = thompson_path

        # Cache for Thompson samples (only used if no external path provided)
        self._cached_thompson_path = None
        self._cached_train_inputs = None

    def _needs_recompute(self) -> bool:
        """Check if cached values need recomputation."""
        current_train_inputs = self.model.train_inputs[0]

        if self._cached_train_inputs is None:
            return True
        if current_train_inputs.shape != self._cached_train_inputs.shape:
            return True
        if not torch.allclose(current_train_inputs, self._cached_train_inputs):
            return True
        return False

    def invalidate_cache(self) -> None:
        """Invalidate cached Thompson path."""
        self._cached_thompson_path = None
        self._cached_train_inputs = None

    def _get_thompson_path(self) -> Callable[[Tensor], Tensor]:
        """Get or create Thompson sample path f ~ P(f|D).

        If an external thompson_path was provided at init, returns that.
        Otherwise, creates and caches a new path.

        The sampling approach depends on self.sample_approach:
        - "ff": Use Fourier features / pathwise sampling (for stationary kernels)
        - "dmvn": Use direct MVN sampling (for any kernel, good for pool-based)
        """
        # Use external path if provided
        if self._external_thompson_path is not None:
            return self._external_thompson_path

        # Use cached path if valid
        if self._cached_thompson_path is not None and not self._needs_recompute():
            return self._cached_thompson_path

        # Create new path
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.sample_approach == "ff":
            # Fourier features / pathwise sampling
            prior_sampler = partial(draw_kernel_feature_paths, num_features=self.num_rff_features)
            sample_path = draw_matheron_paths(
                self.model, sample_shape=torch.Size([1]), prior_sampler=prior_sampler
            )
        else:
            # Direct MVN sampling at reference points
            # Sample once at reference points and create a lookup function
            with torch.no_grad():
                posterior = self.model.posterior(self.reference_points)
                # Sample shape [1] to match pathwise output format
                f_sample = posterior.rsample(sample_shape=torch.Size([1]))  # (1, n, 1)

            # Store the sampled values and reference points for lookup
            sampled_X = self.reference_points.clone()
            sampled_Y = f_sample  # (1, n, 1)

            def sample_path(X: Tensor) -> Tensor:
                """Lookup function values from pre-sampled MVN.

                For points in sampled_X, returns the sampled value.
                For new points, samples from the posterior conditioned on sampled values.
                """
                # Check if X matches reference points (common case for pool-based)
                if X.shape[0] == sampled_X.shape[0] and torch.allclose(X, sampled_X, atol=1e-6):
                    return sampled_Y

                # For arbitrary X, need to sample from joint distribution
                # This is less efficient but handles the general case
                with torch.no_grad():
                    posterior = self.model.posterior(X)
                    return posterior.rsample(sample_shape=torch.Size([1]))

            sample_path = sample_path

        self._cached_thompson_path = sample_path
        self._cached_train_inputs = self.model.train_inputs[0].clone()

        return sample_path

    def _get_thompson_sample_at_reference_points(self) -> Tensor:
        """Get Thompson sample f ~ P(f|D) at integration points."""
        thompson_path = self._get_thompson_path()
        with torch.no_grad():
            f_sample = thompson_path(self.reference_points)
        return f_sample.squeeze(0).squeeze(-1)

    def compute_posterior_stats(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute posterior statistics at arbitrary points.

        Note: This method does NOT wrap with torch.no_grad() to allow gradient
        computation for SMC sampling. Callers should wrap with torch.no_grad()
        when gradients are not needed.

        Args:
            X: (n, d) points to evaluate

        Returns:
            mu: (n,) posterior mean
            sigma: (n,) posterior std

        Implementation Notes:
            We use gpytorch.settings.fast_pred_var() and reset max_eager_kernel_size
            to 512 (GPyTorch's default) for efficient diagonal variance computation.

            Background: BoTorch sets max_eager_kernel_size=4096 globally, which causes
            GPyTorch to eagerly densify the joint covariance matrix when
            n_test + n_train <= 4096 (see exact_prediction_strategies.py:274-277).
            This is O(n^2) and unnecessary when we only need diagonal variances.

            For SMC sampling with 1000-4000 particles, this causes a ~10-40x slowdown.
            By resetting to 512 and using fast_pred_var, GPyTorch uses lazy evaluation
            and only computes the diagonal, which is O(n).

            Standard BoTorch acquisition functions (e.g., qEI, qKG) typically evaluate
            at small batch sizes (q < 100) where eager densification is acceptable.
            SMC sampling evaluates at thousands of points per iteration, making this
            optimization critical.

            Reference: https://github.com/cornellius-gp/gpytorch/discussions/2312
        """
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_eager_kernel_size(512):
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze(-1)
            sigma = posterior.variance.sqrt().squeeze(-1)

        return mu, sigma

    def compute_posterior_stats_at_reference_points(self) -> tuple[Tensor, Tensor]:
        """
        Compute posterior statistics at integration points.

        Returns:
            mu: (n_reference_points,) posterior mean
            sigma: (n_reference_points,) posterior std
        """
        with torch.no_grad():
            return self.compute_posterior_stats(self.reference_points)

    def compute_log_unnorm_density(self, X: Tensor, stabilize: bool = False) -> Tensor:
        """
        Compute log probability log P(x) ∝ log E[exp(λf(x)/T + b(x))] at arbitrary points.

        Note: This method does NOT wrap with torch.no_grad() to allow gradient
        computation for SMC sampling. Callers should wrap with torch.no_grad()
        when gradients are not needed.

        Supports base_approximation modes: None, "mean", "zeroth_order_analytic".

        Args:
            X: (n, d) points to evaluate
            stabilize: If True and using zeroth_order_analytic, subtract max log_prob
                to prevent overflow. Default False (caller handles stabilization).

        Returns:
            log_prob: (n,) unnormalized log probability at each point
        """
        bias = self.bias_fn(X)
        lambda_over_T = self.boltzmann_sign / self.temperature

        # Thompson sampling: log P_f(x) = λf(x)/T + b(x)
        if self.use_thompson_sampling:
            thompson_path = self._get_thompson_path()
            # Note: no torch.no_grad() here; callers decide whether gradients are needed
            f_sample = thompson_path(X).squeeze(0).squeeze(-1)
            log_prob = lambda_over_T * f_sample + bias
            if stabilize:
                log_prob = log_prob - log_prob.max()
            return log_prob

        if self.base_approximation is None:
            # No energy term, just exp(b(x))
            return bias

        elif self.base_approximation == "mean":
            mu, _ = self.compute_posterior_stats(X)
            return lambda_over_T * mu + bias

        elif self.base_approximation == "zeroth_order_analytic":
            # E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)
            mu, sigma = self.compute_posterior_stats(X)
            sigma_sq = sigma ** 2
            log_prob = lambda_over_T * mu + (lambda_over_T ** 2) * sigma_sq / 2 + bias

            if stabilize:
                log_prob = log_prob - log_prob.max()

            return log_prob

        else:
            raise ValueError(f"Unknown base_approximation: {self.base_approximation}")

    def _compute_log_expected_boltzmann_at_reference_points(self) -> Tensor:
        """
        Compute log E[exp(λf(x)/T + b(x))] at each integration point.

        For base_approximation=None:
            log E[...] = b

        For "mean":
            Approximates E[exp(λf/T)] ≈ exp(λμ/T), so log E[...] ≈ λμ/T + b

        For zeroth_order_analytic:
            Uses closed-form MGF of Gaussian:
            E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)
            so log E[...] = λμ/T + (λ/T)²σ²/2 + b

        Returns:
            log_E: (n_reference_points,) log of expected Boltzmann weight
        """
        with torch.no_grad():
            log_E = self.compute_log_unnorm_density(self.reference_points, stabilize=False)

        # Apply stabilization for zeroth_order_analytic at integration points
        if self.stabilize_log_density and self.base_approximation == "zeroth_order_analytic":
            lambda_over_T = self.boltzmann_sign / self.temperature
            _, sigma = self.compute_posterior_stats_at_reference_points()
            sigma_sq = sigma ** 2
            sigma_sq_max = sigma_sq.max()
            log_E = log_E - (lambda_over_T ** 2) * sigma_sq_max / 2

        return log_E

    def _compute_thompson_log_expected_boltzmann_at_reference_points(self) -> Tensor:
        """
        Compute log E[exp(λf(x)/T + b(x))] at integration points using Thompson sampling.

        Draws a single sample f ~ P(f|D) and computes:
            log E[exp(λf/T + b)] ≈ λf/T + b

        This is a single-sample Monte Carlo estimate of the log expected Boltzmann.

        If stabilize_log_density is True, subtracts max to prevent overflow.

        Returns:
            log_E: (n_reference_points,) Monte Carlo estimate of log expected Boltzmann
        """
        f_sample = self._get_thompson_sample_at_reference_points()  # (n,)
        bias = self.bias_fn(self.reference_points)
        lambda_over_T = self.boltzmann_sign / self.temperature
        log_E = lambda_over_T * f_sample + bias

        if self.stabilize_log_density:
            # Subtract max to prevent overflow (preserves ranking)
            log_E = log_E - log_E.max()

        return log_E

    def get_log_expected_boltzmann_at_reference_points(self) -> Tensor:
        """
        Get log E[exp(λf(x)/T + b(x))] at each integration point.

        - use_thompson_sampling=True: Monte Carlo estimate using Thompson sample
        - use_thompson_sampling=False: Deterministic/analytic approximation

        Returns:
            log_E: (n_reference_points,) log expected Boltzmann values
        """
        if self.use_thompson_sampling:
            return self._compute_thompson_log_expected_boltzmann_at_reference_points()
        else:
            return self._compute_log_expected_boltzmann_at_reference_points()

    def get_expected_boltzmann_at_reference_points(self) -> Tensor:
        """
        Get E[exp(λf(x)/T + b(x))] at each integration point.

        For zeroth_order_analytic:
            E[exp(λf/T + b)] = exp(λμ/T + (λ/T)²σ²/2 + b)

        For "mean":
            ≈ exp(λμ/T + b)

        Returns:
            E_boltzmann: (n_reference_points,) expected Boltzmann values

        Note: For numerical stability, prefer compute_integration_weights()
        which uses log-sum-exp internally.

        Raises:
            ValueError: If values would cause exponential overflow.
        """
        log_E = self.get_log_expected_boltzmann_at_reference_points()

        # Check for exponential overflow (float64 max is ~1e308, exp(709) ≈ 8e307)
        max_log = log_E.max().item()
        if max_log > 700:
            raise ValueError(
                f"Exponential overflow detected: max log value = {max_log:.1f} "
                f"(exp(709) overflows float64). This typically happens when using "
                f"'zeroth_order_analytic' base_approximation with large posterior "
                f"variance (σ²/2 term dominates). Solutions:\n"
                f"  1. Use standardized function values (pre-standardize test function)\n"
                f"  2. Set use_outcome_transform=False with pre-standardized Y\n"
                f"  3. Use a different base_approximation (e.g., 'mean')"
            )

        return torch.exp(log_E)

    def compute_integration_weights(self) -> Tensor:
        """
        Compute normalized weights at integration points for weighted expectation.

        Returns P̃(x_i) for each integration point x_i, where:
            P̃(x_i) = E[exp(λf(x_i)/T + b(x_i))] / Σ_j E[exp(λf(x_j)/T + b(x_j))]

        These weights are used to compute expectations like E_{P̃}[σ²(x)].

        Uses the log-sum-exp trick for numerical stability.

        Returns:
            weights: (n_reference_points,) normalized weights that sum to 1
        """
        log_E = self.get_log_expected_boltzmann_at_reference_points()
        # Use log-sum-exp trick: P = exp(log_E - logsumexp(log_E))
        log_Z = torch.logsumexp(log_E, dim=0)
        return torch.exp(log_E - log_Z)

    @property
    def posterior_stats_at_reference_points(self) -> dict[str, Tensor]:
        """Get posterior statistics at integration points as a dict."""
        mu, sigma = self.compute_posterior_stats_at_reference_points()
        return {'mu': mu, 'sigma': sigma}


class _SID_iVAR_Base(VarianceConstraintMixin, FantasyVarianceMixin, AnalyticAcquisitionFunction):
    """
    Base class for AB-SID-iVAR acquisition functions.

    This is an internal base class providing shared functionality for the
    specialized discrete and continuous variants. Do not instantiate directly.

    Implements (PDF Eq. 11):
        x_t = arg min_{x ∈ X̄_t} E_{P̃_{t-1}}[σ²_t(x*) | x_t = x]

    where P̃_t is the self-induced distribution (SID):
        P̃_t(x) ∝ exp(λ · g(x) / T + b(x))

    Subclasses handle the difference in how E_P[σ²] is computed:
    - Discrete: Grid points + P-weights (importance sampling)
    - Continuous: SMC samples from P + equal weights (direct Monte Carlo)

    Note:
        If function output range exceeds ~700, this may cause numerical issues
        due to exponential underflow/overflow in Boltzmann weight computation.
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        base_approximation: Optional[str] = None,
        use_thompson_sampling: bool = False,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
        sample_approach: str = "ff",
    ):
        """
        Args:
            space: Design space (discrete or continuous).
            model: Fitted SingleTaskGP model
            bias_fn: Bias function b(x). If None, uses zero bias.
            temperature: Temperature parameter for Boltzmann density (default: 1.0)
            boltzmann_sign: Sign λ in the Boltzmann distribution (default: -1.0)
            seed: Random seed for reproducibility
            constraint_type: Type of constraint set definition, or None to disable constraints.
            base_approximation: Deterministic/analytic base approximation for energy.
            use_thompson_sampling: If True, use Thompson sampling for energy.
            stabilize_log_density: If True, stabilize log density to prevent overflow.
            num_rff_features: Number of RFF features for Thompson sampling. Default: 4096.
            sample_approach: Approach for Thompson sampling: "ff" (Fourier features) or
                "dmvn" (direct MVN sampling). Default: "ff".
        """
        super().__init__(model=model)

        self.space = space
        self.seed = seed
        self.base_approximation = base_approximation
        self.use_thompson_sampling = use_thompson_sampling
        self.num_rff_features = num_rff_features
        self.sample_approach = sample_approach

        # Cache for tracking model changes (for continuous space reference point updates)
        self._cached_reference_train_inputs = None

        # Store energy distribution parameters for recreating after reference point updates
        self._energy_params = {
            "temperature": temperature,
            "boltzmann_sign": boltzmann_sign,
            "bias_fn": bias_fn,
            "stabilize_log_density": stabilize_log_density,
        }

        # Initialize reference points (subclasses set this)
        self._reference_points = None
        self._energy_dist = None

        # Initialize constraint set from mixin
        self._init_constraint(constraint_type)

    def _create_energy_distribution(
        self,
        base_approximation: Optional[str] = None,
        use_thompson_sampling: Optional[bool] = None,
        thompson_path: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> BoltzmannEnergyDistribution:
        """Create BoltzmannEnergyDistribution with current reference points.

        Args:
            base_approximation: Override base_approximation (for SMC sampling).
            use_thompson_sampling: Override use_thompson_sampling (for SMC sampling).
            thompson_path: External Thompson path (for TS-SID-iVAR to share path
                between SMC and acquisition function).
        """
        return BoltzmannEnergyDistribution(
            model=self.model,
            reference_points=self._reference_points,
            temperature=self._energy_params["temperature"],
            boltzmann_sign=self._energy_params["boltzmann_sign"],
            base_approximation=base_approximation if base_approximation is not None else self.base_approximation,
            use_thompson_sampling=use_thompson_sampling if use_thompson_sampling is not None else self.use_thompson_sampling,
            bias_fn=self._energy_params["bias_fn"],
            seed=self.seed,
            stabilize_log_density=self._energy_params["stabilize_log_density"],
            num_rff_features=self.num_rff_features,
            thompson_path=thompson_path,
            sample_approach=self.sample_approach,
        )

    @property
    def reference_points(self) -> Tensor:
        """Get reference points."""
        return self._reference_points

    @property
    def target_distribution(self) -> Tensor:
        """Get the target distribution P̃ at reference points."""
        return self._energy_dist.compute_integration_weights()

    @property
    def posterior_stats_at_reference_points(self) -> dict[str, Tensor]:
        """Get posterior statistics at reference points."""
        return self._energy_dist.posterior_stats_at_reference_points

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate acquisition function. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")


class AB_SID_iVAR_Discrete(_SID_iVAR_Base):
    """
    AB-SID-iVAR for discrete design spaces.

    Analytic Boltzmann (AB) uses the zeroth-order analytic approximation:
        P(x) ∝ E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)

    For discrete spaces:
    - Reference points = grid/candidates from space
    - E_P[σ²] = Σ P(x_i) · σ²(x_i) (importance sampling with P-weights)

    This is the correct approach for discrete grids where we cannot sample from P.
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
    ):
        if not space.is_discrete:
            raise ValueError("AB_SID_iVAR_Discrete requires a discrete DesignSpace")

        super().__init__(
            space=space,
            model=model,
            bias_fn=bias_fn,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            seed=seed,
            constraint_type=constraint_type,
            base_approximation="zeroth_order_analytic",
            use_thompson_sampling=False,
            stabilize_log_density=stabilize_log_density,
            num_rff_features=num_rff_features,
        )

        # Set reference points from space candidates
        self._reference_points = space.candidates
        self._energy_dist = self._create_energy_distribution()

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate AB-SID-iVAR for discrete spaces.

        Uses importance sampling: E_P[σ²] = Σ P(x_i) · σ²(x_i)
        where P(x) = exp(λμ/T + (λ/T)²σ²/2) / Z

        Args:
            X: (batch, q=1, d) or (batch, d) candidate points

        Returns:
            acq: (batch,) acquisition values (negative expected variance)
        """
        if X.dim() == 3:
            X = X.squeeze(1)

        # Get UNNORMALIZED density weights for expectation (importance sampling)
        # Using unnormalized weights is ranking-equivalent for argmin
        unnorm_weights = self._energy_dist.get_expected_boltzmann_at_reference_points()

        # Prepare cache and compute fantasy variance
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_fantasy = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        # Weighted sum (importance sampling): E_P[σ²] ∝ Σ P(x_i) · σ²(x_i)
        weighted_var = (unnorm_weights.unsqueeze(0) * var_fantasy).sum(dim=1)

        # Apply constraint set filtering
        acq = self._apply_constraint_to_acquisition(X, -weighted_var)
        return acq


class AB_SID_iVAR_Continuous(_SID_iVAR_Base):
    """
    AB-SID-iVAR for continuous design spaces.

    Analytic Boltzmann (AB) uses the zeroth-order analytic approximation:
        P(x) ∝ E[exp(λf/T)] = exp(λμ/T + (λ/T)²σ²/2)

    For continuous spaces:
    - Reference points = SMC samples from P (zeroth_order_analytic distribution)
    - E_P[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P (direct Monte Carlo, EQUAL weights)

    This fixes the double-weighting bug in the old implementation where SMC
    samples from P were also weighted by P in forward().

    Key insight: When x_i ~ P, the correct estimator for E_P[f(x)] is simply
    (1/N) Σ f(x_i), NOT (1/N) Σ P(x_i) · f(x_i).
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
    ):
        if space.is_discrete:
            raise ValueError("AB_SID_iVAR_Continuous requires a continuous DesignSpace")

        super().__init__(
            space=space,
            model=model,
            bias_fn=bias_fn,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            seed=seed,
            constraint_type=constraint_type,
            base_approximation="zeroth_order_analytic",
            use_thompson_sampling=False,
            stabilize_log_density=stabilize_log_density,
            num_rff_features=num_rff_features,
        )

        # Reference points will be generated on first forward() call
        self._reference_points = None

    def _reference_points_need_update(self) -> bool:
        """Check if reference points need regeneration."""
        current_train_inputs = self.model.train_inputs[0]
        if self._cached_reference_train_inputs is None:
            return True
        if current_train_inputs.shape != self._cached_reference_train_inputs.shape:
            return True
        if not torch.allclose(current_train_inputs, self._cached_reference_train_inputs):
            return True
        return False

    def _ensure_reference_points_updated(self) -> None:
        """Update reference points if model changed.

        For AB-SID-iVAR, SMC samples from the ANALYTIC Boltzmann distribution:
            P_analytic(x) ∝ exp(λμ(x)/T + (λ/T)²σ²(x)/2)

        This uses BoltzmannEnergyDistribution with base_approximation="zeroth_order_analytic".

        This is different from TS-SID-iVAR which samples from the Thompson-induced
        distribution P_f(x) ∝ exp(λf(x)/T) using ThompsonSMCEnergyDistribution.

        See ThompsonSMCEnergyDistribution class docstring for detailed comparison.
        """
        if not self._reference_points_need_update():
            return

        # Create ANALYTIC energy distribution for SMC sampling
        # AB uses zeroth_order_analytic: log P = λμ/T + (λ/T)²σ²/2
        temp_energy_dist = BoltzmannEnergyDistribution(
            model=self.model,
            reference_points=torch.empty(0, self.space.dim),  # placeholder
            temperature=self._energy_params["temperature"],
            boltzmann_sign=self._energy_params["boltzmann_sign"],
            base_approximation="zeroth_order_analytic",  # AB uses analytic approximation
            use_thompson_sampling=False,  # NOT Thompson sampling
            bias_fn=self._energy_params["bias_fn"],
            seed=self.seed,
            stabilize_log_density=False,  # SMC handles its own stability
        )

        # Generate reference points via SMC from P_analytic
        self._reference_points = self.space.update_reference_points(
            energy_distribution=temp_energy_dist
        )

        # Store SMC info for timing analysis
        if hasattr(self.space, 'last_generator_info') and self.space.last_generator_info is not None:
            self._last_smc_info = self.space.last_generator_info

        # Recreate energy distribution with new reference points
        self._energy_dist = self._create_energy_distribution()

        # Update cache
        self._cached_reference_train_inputs = self.model.train_inputs[0].clone()

    @property
    def reference_points(self) -> Tensor:
        """Get reference points, updating if necessary."""
        self._ensure_reference_points_updated()
        return self._reference_points

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate AB-SID-iVAR for continuous spaces.

        Uses direct Monte Carlo: E_P[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P
        Reference points are SMC samples from P, so we use EQUAL weights.

        Args:
            X: (batch, q=1, d) or (batch, d) candidate points

        Returns:
            acq: (batch,) acquisition values (negative mean variance)
        """
        if X.dim() == 3:
            X = X.squeeze(1)

        # Ensure reference points are up-to-date
        self._ensure_reference_points_updated()

        # Prepare cache and compute fantasy variance
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_fantasy = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        # EQUAL weights: E_P[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P
        # Since reference points are SMC samples from P, no reweighting needed
        mean_var = var_fantasy.mean(dim=1)

        return -mean_var


class Mu_SID_iVAR_Continuous(AB_SID_iVAR_Continuous):
    """
    Mu-SID-iVAR: ablation using only posterior mean exp(λμ/T).

    Identical to AB-SID-iVAR but uses base_approximation="mean" instead of
    "zeroth_order_analytic", to show importance of the variance term.
    """

    def _ensure_reference_points_updated(self) -> None:
        """Update reference points using mean-only distribution."""
        if not self._reference_points_need_update():
            return

        temp_energy_dist = BoltzmannEnergyDistribution(
            model=self.model,
            reference_points=torch.empty(0, self.space.dim),
            temperature=self._energy_params["temperature"],
            boltzmann_sign=self._energy_params["boltzmann_sign"],
            base_approximation="mean",  # Only difference from AB-SID
            use_thompson_sampling=False,
            bias_fn=self._energy_params["bias_fn"],
            seed=self.seed,
            stabilize_log_density=False,
        )

        self._reference_points = self.space.update_reference_points(
            energy_distribution=temp_energy_dist
        )

        if hasattr(self.space, 'last_generator_info') and self.space.last_generator_info is not None:
            self._last_smc_info = self.space.last_generator_info

        self._energy_dist = self._create_energy_distribution()
        self._cached_reference_train_inputs = self.model.train_inputs[0].clone()


class Mu_SID_EPIG_Continuous(Mu_SID_iVAR_Continuous):
    """
    Mu-SID-EPIG: EPIG with mean-only target weighting (no uncertainty truncation).

    Reference points x* are sampled via SMC from exp(λμ/T) (inherited from
    Mu_SID_iVAR_Continuous). The acquisition replaces the iVAR objective with
    the EPIG information-gain objective:

        Mu-SID-EPIG(x) = (1/N) Σ 0.5 * log(σ²_before(x_i) / σ²_after(x_i, x))

    where x_i ~ exp(λμ/T). This is the natural EPIG baseline using GP
    predictive mean as the surrogate target measure.
    """

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 3:
            X = X.squeeze(1)

        self._ensure_reference_points_updated()

        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_before = cache.var_before
        var_after = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        eps = 1e-10
        log_ratio = torch.log((var_before + eps) / (var_after + eps))
        return 0.5 * log_ratio.mean(dim=-1)


class Mu_SID_EPIG_Discrete(AB_SID_iVAR_Discrete):
    """
    Mu-SID-EPIG for discrete (pool-based) design spaces.

    Reference points are the pool/candidates with mean-only Boltzmann weighting
    exp(λμ/T) (no uncertainty term). Acquisition uses the EPIG information-gain
    objective rather than expected-variance reduction:

        Mu-SID-EPIG(x) = Σ_i w(x_i) * 0.5 * log(σ²_before(x_i) / σ²_after(x_i, x))

    where w(x_i) ∝ exp(λμ(x_i)/T) over the pool. This is the natural EPIG
    baseline for pool-based AL using GP predictive mean as the target measure.
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
    ):
        if not space.is_discrete:
            raise ValueError("Mu_SID_EPIG_Discrete requires a discrete DesignSpace")

        # Skip AB_SID_iVAR_Discrete.__init__ to override base_approximation
        _SID_iVAR_Base.__init__(
            self,
            space=space,
            model=model,
            bias_fn=bias_fn,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            seed=seed,
            constraint_type=constraint_type,
            base_approximation="mean",  # mean-only: exp(λμ/T)
            use_thompson_sampling=False,
            stabilize_log_density=False,  # mean-only doesn't support stabilization
            num_rff_features=num_rff_features,
        )

        self._reference_points = space.candidates
        self._energy_dist = self._create_energy_distribution()

    def forward(self, X: Tensor) -> Tensor:
        if X.dim() == 3:
            X = X.squeeze(1)

        unnorm_weights = self._energy_dist.get_expected_boltzmann_at_reference_points()

        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_before = cache.var_before
        var_after = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        eps = 1e-10
        info_gain = 0.5 * torch.log((var_before + eps) / (var_after + eps))
        weighted_info = (unnorm_weights.unsqueeze(0) * info_gain).sum(dim=1)

        acq = self._apply_constraint_to_acquisition(X, weighted_info)
        return acq


class TS_SID_iVAR_Discrete(_SID_iVAR_Base):
    """
    TS-SID-iVAR for discrete design spaces.

    Thompson Sampling (TS) uses a sample from the GP posterior:
        f ~ P(f|D), then P_f(x) ∝ exp(λf(x)/T)

    For discrete spaces:
    - Reference points = grid/candidates from space
    - E_{P_f}[σ²] = Σ P_f(x_i) · σ²(x_i) (importance sampling with P_f-weights)

    This is the correct approach for discrete grids where we cannot sample from P_f.
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
        sample_approach: str = "ff",
    ):
        if not space.is_discrete:
            raise ValueError("TS_SID_iVAR_Discrete requires a discrete DesignSpace")

        super().__init__(
            space=space,
            model=model,
            bias_fn=bias_fn,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            seed=seed,
            constraint_type=constraint_type,
            base_approximation=None,  # Not used when use_thompson_sampling=True
            use_thompson_sampling=True,
            stabilize_log_density=stabilize_log_density,
            num_rff_features=num_rff_features,
            sample_approach=sample_approach,
        )

        # Set reference points from space candidates
        self._reference_points = space.candidates
        self._energy_dist = self._create_energy_distribution()

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate TS-SID-iVAR for discrete spaces.

        Uses importance sampling: E_{P_f}[σ²] = Σ P_f(x_i) · σ²(x_i)
        where P_f(x) = exp(λf(x)/T) / Z and f ~ P(f|D)

        Args:
            X: (batch, q=1, d) or (batch, d) candidate points

        Returns:
            acq: (batch,) acquisition values (negative expected variance)
        """
        if X.dim() == 3:
            X = X.squeeze(1)

        # Get UNNORMALIZED density weights from Thompson sample
        unnorm_weights = self._energy_dist.get_expected_boltzmann_at_reference_points()

        # Prepare cache and compute fantasy variance
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_fantasy = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        # Weighted sum (importance sampling): E_{P_f}[σ²] ∝ Σ P_f(x_i) · σ²(x_i)
        weighted_var = (unnorm_weights.unsqueeze(0) * var_fantasy).sum(dim=1)

        # Apply constraint set filtering
        acq = self._apply_constraint_to_acquisition(X, -weighted_var)
        return acq


class HAL_Acquisition:
    """
    HAL-style acquisition using single-chain MCMC from LCB-biased distribution.

    Implements the Hyperactive Learning (HAL) approach for continuous domains:
    1. Run single MCMC chain from biased distribution:
       P_HAL(x) ∝ exp((boltzmann_sign · μ + τσ) / T)
       - boltzmann_sign = -1 (λ < 0): bias toward low μ (original HAL for minimization)
       - boltzmann_sign = +1 (λ > 0): bias toward high μ (for maximization / exp(λf))
    2. At each step, check if uncertainty σ exceeds threshold
    3. If threshold exceeded → STOP and return that point (reactive selection)
    4. If max steps reached → return point with highest uncertainty seen

    This mimics HAL's biased MD approach where the simulation is stopped
    when the model uncertainty becomes too high (indicating need for retraining).
    Uses the same boltzmann_sign convention as AB-SID-iVAR: when specifying
    boltzmann_lambda in the config, sign(λ) → boltzmann_sign, 1/|λ| → temperature.

    Key differences from SID-iVAR:
    - HAL: Single chain, reactive threshold-based stopping
    - SID-iVAR: Optimization-based, proactive variance reduction

    Reference: van der Oord et al., "Hyperactive learning for data-driven
    interatomic potentials", npj Computational Materials (2023)

    Note: This class is NOT an AcquisitionFunction subclass because HAL
    doesn't fit the standard "evaluate everywhere, optimize" pattern.
    Use via run_hal_experiment() or call select_next_point() directly.
    """

    def __init__(
        self,
        model: Model,
        bounds: Tensor,
        tau: float = 1.0,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        threshold: float = 0.1,
        max_mcmc_steps: int = 2000,
        burn_in_steps: int = 50,
        step_size: float = 0.1,
        seed: Optional[int] = None,
        train_X: Optional[Tensor] = None,
        init_strategy: str = "random",
    ):
        """
        Args:
            model: Fitted GP model
            bounds: (2, d) tensor of bounds
            tau: Exploration parameter τ. Higher = more exploration.
            temperature: Temperature T for the biased distribution.
                When using boltzmann_lambda, this is set to 1/|λ|.
            boltzmann_sign: Sign of the Boltzmann parameter (+1 or -1).
                +1 (λ > 0): bias toward high μ (maximize, e.g., exp(λf) with λ > 0)
                -1 (λ < 0): bias toward low μ (minimize, e.g., exp(λf) with λ < 0)
                Same convention as AB-SID-iVAR's boltzmann_sign.
            train_X: Training data points. Required when init_strategy="from_data".
            init_strategy: How to initialize the MCMC chain.
                "random": uniform random in bounds (default, uses burn-in)
                "from_data": random training point (infeasible region, no burn-in)
            threshold: Uncertainty threshold σ for reactive selection.
                If σ > threshold, stop and select that point.
                If no point exceeds threshold during chain, selects highest σ seen.
                With standardized GP, σ ≈ 1 in explored regions, so threshold > 1 is typical.
            max_mcmc_steps: Maximum MCMC steps before stopping
            burn_in_steps: Number of burn-in steps before checking threshold.
                During burn-in, chain equilibrates to P_HAL without triggering.
                Similar to ACEHAL's temperature ramping for thermalization.
            step_size: Step size for random walk proposals (fraction of domain)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.bounds = bounds
        self.tau = tau
        self.temperature = temperature
        self.boltzmann_sign = boltzmann_sign
        self.threshold = threshold
        self.max_mcmc_steps = max_mcmc_steps
        self.burn_in_steps = burn_in_steps
        self.train_X = train_X
        self.init_strategy = init_strategy
        self.step_size = step_size
        self.seed = seed

        # Will be populated during MCMC
        self.chain_history = None
        self.selected_point = None

    def _compute_log_density(self, X: Tensor) -> Tensor:
        """
        Compute log P_HAL(x) = (boltzmann_sign · μ + τσ) / T.

        boltzmann_sign = +1 (λ > 0): bias toward high μ (maximize)
        boltzmann_sign = -1 (λ < 0): bias toward low μ (minimize)

        Args:
            X: (n, d) or (d,) points to evaluate

        Returns:
            log_prob: (n,) or scalar unnormalized log probability
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze(-1)
            sigma = posterior.variance.sqrt().squeeze(-1)

        # log P = (boltzmann_sign · μ + τ · σ) / T
        # boltzmann_sign = +1: bias toward high μ (maximize)
        # boltzmann_sign = -1: bias toward low μ (minimize, original HAL)
        log_prob = (self.boltzmann_sign * mu + self.tau * sigma) / self.temperature

        return log_prob.squeeze() if log_prob.numel() == 1 else log_prob

    def _compute_uncertainty(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute uncertainty σ(x) at point X.

        Returns:
            sigma: posterior std (used for threshold comparison)
            mu: posterior mean
            sigma: posterior std (same as first return)
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        with torch.no_grad():
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze()
            sigma = posterior.variance.sqrt().squeeze()

        return sigma, mu, sigma

    def select_next_point(self) -> tuple[Tensor, dict]:
        """
        Run HAL-style single-chain MCMC with threshold stopping.

        The chain explores P_HAL(x) ∝ exp(-(μ - τσ)) and stops when
        uncertainty σ exceeds threshold.

        Returns:
            selected_point: (1, d) tensor of selected point
            info: dict with selection details
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dim = self.bounds.shape[1]
        dtype = self.bounds.dtype
        device = self.bounds.device

        # Initialize chain
        if self.init_strategy == "from_data" and self.train_X is not None:
            # Start from a random training point (infeasible region: σ ≈ 0)
            # Chain walks outward until σ > threshold — no burn-in needed
            idx = torch.randint(len(self.train_X), (1,)).item()
            x_current = self.train_X[idx].clone().to(dtype=dtype, device=device)
            effective_burn_in = 0
        else:
            # Uniform random in bounds (original behavior, uses burn-in)
            x_current = torch.rand(dim, dtype=dtype, device=device)
            x_current = self.bounds[0] + x_current * (self.bounds[1] - self.bounds[0])
            effective_burn_in = self.burn_in_steps

        log_p_current = self._compute_log_density(x_current)
        sigma_current, mu_current, _ = self._compute_uncertainty(x_current)

        # Track best point seen (highest σ)
        best_sigma = sigma_current.item()
        best_point = x_current.clone()
        best_mu = mu_current.item()

        # Chain history for diagnostics
        n_accepted = 0
        step_counter = 0
        triggered_threshold = False

        # Scale step size by domain size
        domain_size = self.bounds[1] - self.bounds[0]

        for step in range(self.max_mcmc_steps):
            step_counter += 1

            # Check threshold AFTER burn-in (reactive stopping like HAL)
            # With from_data init, burn-in is 0 — check immediately
            if step >= effective_burn_in:
                if sigma_current.item() > self.threshold:
                    triggered_threshold = True
                    break

            # Random walk proposal
            proposal = x_current + self.step_size * domain_size * torch.randn(dim, dtype=dtype, device=device)

            # Reflect at boundaries
            proposal = torch.max(proposal, self.bounds[0])
            proposal = torch.min(proposal, self.bounds[1])

            # Metropolis-Hastings acceptance
            log_p_proposal = self._compute_log_density(proposal)
            log_alpha = log_p_proposal - log_p_current

            if torch.log(torch.rand(1, dtype=dtype, device=device)) < log_alpha:
                x_current = proposal
                log_p_current = log_p_proposal
                sigma_current, mu_current, _ = self._compute_uncertainty(x_current)
                n_accepted += 1

                # Update best if this is highest uncertainty seen
                if sigma_current.item() > best_sigma:
                    best_sigma = sigma_current.item()
                    best_point = x_current.clone()
                    best_mu = mu_current.item()

        # Select point
        if triggered_threshold:
            # Threshold was exceeded - return current point
            self.selected_point = x_current.unsqueeze(0)
            selection_method = "threshold"
            selected_sigma = sigma_current.item()
            selected_mu = mu_current.item()
        else:
            # Max steps reached - return best point seen (highest σ)
            self.selected_point = best_point.unsqueeze(0)
            selection_method = "max_uncertainty"
            selected_sigma = best_sigma
            selected_mu = best_mu

        info = {
            "tau": self.tau,
            "threshold": self.threshold,
            "max_mcmc_steps": self.max_mcmc_steps,
            "burn_in_steps": effective_burn_in,
            "init_strategy": self.init_strategy,
            "steps_taken": step_counter,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / step_counter if step_counter > 0 else 0.0,
            "selection_method": selection_method,
            "triggered_threshold": triggered_threshold,
            "selected_sigma": selected_sigma,
            "selected_mu": selected_mu,
            "best_sigma_seen": best_sigma,
        }

        return self.selected_point, info


class TS_SID_iVAR_Continuous(_SID_iVAR_Base):
    """
    TS-SID-iVAR for continuous design spaces.

    Thompson Sampling (TS) uses a sample from the GP posterior:
        f ~ P(f|D), then P_f(x) ∝ exp(λf(x)/T)

    For continuous spaces:
    - Reference points = SMC samples from P_f (Thompson-induced Boltzmann)
    - E_{P_f}[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P_f (direct Monte Carlo, EQUAL weights)

    This fixes the bug in the old implementation where SMC sampled from uniform
    instead of from the Thompson-induced distribution P_f.

    Key insight: The whole point of Thompson sampling SID is that the target
    distribution is characterized by the Thompson sample f, so SMC should
    sample from exp(λf/T), not from uniform.
    """

    def __init__(
        self,
        space: DesignSpace,
        model: SingleTaskGP,
        bias_fn: Optional[Callable[[Tensor], Tensor]] = None,
        temperature: float = 1.0,
        boltzmann_sign: float = -1.0,
        seed: Optional[int] = None,
        constraint_type: Optional[str] = None,
        stabilize_log_density: bool = True,
        num_rff_features: int = 2048,
        sample_approach: str = "ff",
    ):
        if space.is_discrete:
            raise ValueError("TS_SID_iVAR_Continuous requires a continuous DesignSpace")

        super().__init__(
            space=space,
            model=model,
            bias_fn=bias_fn,
            temperature=temperature,
            boltzmann_sign=boltzmann_sign,
            seed=seed,
            constraint_type=constraint_type,
            base_approximation=None,  # Not used when use_thompson_sampling=True
            use_thompson_sampling=True,
            stabilize_log_density=stabilize_log_density,
            num_rff_features=num_rff_features,
            sample_approach=sample_approach,
        )

        # Reference points will be generated on first forward() call
        self._reference_points = None

        # Cache for Thompson path (shared between SMC and forward)
        self._thompson_path = None
        self._thompson_path_train_inputs = None

    def _get_or_create_thompson_path(self):
        """Get or create a Thompson sample path f ~ P(f|D)."""
        current_train_inputs = self.model.train_inputs[0]

        # Check if we need to regenerate
        if self._thompson_path is not None:
            if self._thompson_path_train_inputs is not None:
                if current_train_inputs.shape == self._thompson_path_train_inputs.shape:
                    if torch.allclose(current_train_inputs, self._thompson_path_train_inputs):
                        return self._thompson_path

        # Generate new Thompson path
        if self.seed is not None:
            torch.manual_seed(self.seed)

        prior_sampler = partial(draw_kernel_feature_paths, num_features=self.num_rff_features)
        self._thompson_path = draw_matheron_paths(
            self.model, sample_shape=torch.Size([1]), prior_sampler=prior_sampler
        )
        self._thompson_path_train_inputs = current_train_inputs.clone()

        return self._thompson_path

    def _reference_points_need_update(self) -> bool:
        """Check if reference points need regeneration."""
        current_train_inputs = self.model.train_inputs[0]
        if self._cached_reference_train_inputs is None:
            return True
        if current_train_inputs.shape != self._cached_reference_train_inputs.shape:
            return True
        if not torch.allclose(current_train_inputs, self._cached_reference_train_inputs):
            return True
        return False

    def _ensure_reference_points_updated(self) -> None:
        """Update reference points if model changed.

        For TS-SID-iVAR, SMC samples from the Thompson-induced distribution:
            P_f(x) ∝ exp(λf(x)/T)

        where f ~ P(f|D) is a sample from the GP posterior.

        This is different from AB-SID-iVAR which samples from the analytic distribution:
            P_analytic(x) ∝ exp(λμ(x)/T + (λ/T)²σ²(x)/2)

        Both use BoltzmannEnergyDistribution, but with different settings:
        - AB: base_approximation="zeroth_order_analytic", use_thompson_sampling=False
        - TS: base_approximation=None, use_thompson_sampling=True, thompson_path=<path>
        """
        if not self._reference_points_need_update():
            return

        # Get or create Thompson path (pathwise sample from GP posterior)
        thompson_path = self._get_or_create_thompson_path()

        # Create Thompson-based energy distribution for SMC sampling
        # Pass the external thompson_path so SMC uses the same sample as forward()
        thompson_energy_dist = BoltzmannEnergyDistribution(
            model=self.model,
            reference_points=torch.empty(0, self.space.dim),  # placeholder
            temperature=self._energy_params["temperature"],
            boltzmann_sign=self._energy_params["boltzmann_sign"],
            base_approximation=None,  # Not used when use_thompson_sampling=True
            use_thompson_sampling=True,  # TS uses Thompson sampling
            bias_fn=self._energy_params["bias_fn"],
            seed=self.seed,
            stabilize_log_density=False,  # SMC handles its own stability
            thompson_path=thompson_path,  # Pass external path for consistency
        )

        # Generate reference points via SMC from P_f (Thompson-induced distribution)
        self._reference_points = self.space.update_reference_points(
            energy_distribution=thompson_energy_dist
        )

        # Store SMC info for timing analysis
        if hasattr(self.space, 'last_generator_info') and self.space.last_generator_info is not None:
            self._last_smc_info = self.space.last_generator_info

        # Recreate energy distribution with new reference points
        # Pass the same thompson_path so weights are computed with same sample as SMC
        self._energy_dist = self._create_energy_distribution(thompson_path=thompson_path)

        # Update cache
        self._cached_reference_train_inputs = self.model.train_inputs[0].clone()

    @property
    def reference_points(self) -> Tensor:
        """Get reference points, updating if necessary."""
        self._ensure_reference_points_updated()
        return self._reference_points

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate TS-SID-iVAR for continuous spaces.

        Uses direct Monte Carlo: E_{P_f}[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P_f
        Reference points are SMC samples from P_f, so we use EQUAL weights.

        Args:
            X: (batch, q=1, d) or (batch, d) candidate points

        Returns:
            acq: (batch,) acquisition values (negative mean variance)
        """
        if X.dim() == 3:
            X = X.squeeze(1)

        # Ensure reference points are up-to-date
        self._ensure_reference_points_updated()

        # Prepare cache and compute fantasy variance
        cache = self._prepare_fantasy_variance_cache(apply_output_transform=False)
        var_fantasy = self._compute_fantasy_variance(X, cache, apply_output_transform=False)

        # EQUAL weights: E_{P_f}[σ²] = (1/N) Σ σ²(x_i) where x_i ~ P_f
        # Since reference points are SMC samples from P_f, no reweighting needed
        mean_var = var_fantasy.mean(dim=1)

        return -mean_var

