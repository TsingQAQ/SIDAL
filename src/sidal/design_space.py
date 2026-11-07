from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Dict

import torch


class DesignSpace(ABC):
    """Abstract base class for design spaces used in active learning."""

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Whether the design space is discrete."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the design space."""
        pass


class DiscreteDesignSpace(DesignSpace):
    """A discrete design space defined by a set of candidate points."""

    def __init__(self, candidates: torch.Tensor):
        """
        Args:
            candidates: A tensor of shape (N, d) representing N candidate points in d dimensions.
        """
        self.candidates = candidates

    @property
    def is_discrete(self) -> bool:
        """Whether the design space is discrete."""
        return True

    @property
    def dim(self) -> int:
        """Dimensionality of the design space."""
        return self.candidates.shape[-1]
    

class GridDiscreteDesignSpace(DiscreteDesignSpace):
    """A discrete design space defined by a grid within given bounds."""

    def __init__(
        self,
        bounds: torch.Tensor,
        n_grid_per_dim: int,
    ):
        """
        Args:
            bounds: A tensor of shape (2, d) representing the lower and upper bounds of the design space.
            n_grid_per_dim: The number of grid points per dimension.
        """
        d = bounds.shape[1]
        grids = [
            torch.linspace(bounds[0, i], bounds[1, i], n_grid_per_dim, dtype=bounds.dtype)
            for i in range(d)
        ]
        mesh = torch.meshgrid(*grids, indexing='ij')
        candidates = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        super().__init__(candidates=candidates)


class ContinuousDesignSpace(DesignSpace):
    """A continuous design space with dynamically generated reference points.

    For continuous spaces, reference points are generated via a sampling method
    (e.g., SMC) from the target distribution. These points are used for
    integration in acquisition functions like AB-SID-iVAR.

    Unlike discrete spaces where candidates = reference points, continuous spaces
    separate:
    - Candidate search: continuous optimization via L-BFGS over bounds
    - Reference points: samples from target distribution for integration
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        reference_point_generator: Optional[Callable[..., torch.Tensor]] = None,
        reference_point_generator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            bounds: A tensor of shape (2, d) representing the lower and upper bounds.
            reference_point_generator: A callable that generates reference points.
                Signature: generator(energy_distribution, bounds, **kwargs) -> Tensor
                If None, reference points must be set manually via set_reference_points().
            reference_point_generator_kwargs: Keyword arguments passed to the generator.
        """
        self.bounds = bounds
        self._reference_point_generator = reference_point_generator
        self._reference_point_generator_kwargs = reference_point_generator_kwargs or {}
        self._reference_points: Optional[torch.Tensor] = None
        self._last_generator_info: Optional[Dict[str, Any]] = None  # SMC timing info

    @property
    def is_discrete(self) -> bool:
        """Whether the design space is discrete."""
        return False

    @property
    def dim(self) -> int:
        """Dimensionality of the design space."""
        return self.bounds.shape[-1]

    @property
    def reference_points(self) -> torch.Tensor:
        """Get reference points for integration.

        Raises:
            RuntimeError: If reference points have not been generated yet.
        """
        if self._reference_points is None:
            raise RuntimeError(
                "Reference points have not been generated. Call "
                "update_reference_points() first."
            )
        return self._reference_points

    def set_reference_points(self, points: torch.Tensor) -> None:
        """Manually set reference points.

        Args:
            points: (n, d) tensor of reference points.
        """
        self._reference_points = points

    @property
    def last_generator_info(self) -> Optional[Dict[str, Any]]:
        """Get info from the last reference point generation (e.g., SMC timing)."""
        return self._last_generator_info

    def update_reference_points(
        self,
        energy_distribution: Any,
        **kwargs,
    ) -> torch.Tensor:
        """Generate new reference points using the configured generator.

        This should be called once per AL iteration after the model is updated.

        Args:
            energy_distribution: A BoltzmannEnergyDistribution instance that
                provides compute_log_unnorm_density() for sampling.
            **kwargs: Additional keyword arguments passed to the generator,
                overriding those in reference_point_generator_kwargs.

        Returns:
            reference_points: (n, d) tensor of generated reference points.

        Raises:
            RuntimeError: If no reference point generator was configured.
        """
        if self._reference_point_generator is None:
            raise RuntimeError(
                "No reference_point_generator configured. Either pass one to "
                "__init__ or use set_reference_points() to set points manually."
            )

        # Merge kwargs: init kwargs as base, method kwargs override
        all_kwargs = {**self._reference_point_generator_kwargs, **kwargs}

        # Always request info from the generator if it supports it
        all_kwargs["return_info"] = True

        result = self._reference_point_generator(
            energy_distribution=energy_distribution,
            bounds=self.bounds,
            **all_kwargs,
        )

        # Handle generators that return (points, info) tuple
        if isinstance(result, tuple) and len(result) == 2:
            self._reference_points, self._last_generator_info = result
        else:
            self._reference_points = result
            self._last_generator_info = None

        return self._reference_points

    def sample(self, n: int, sample_method: str = "uniform") -> torch.Tensor:
        """Sample n points uniformly from the continuous space.

        Args:
            n: Number of points to sample.
            sample_method: Sampling method (currently only "uniform" supported).

        Returns:
            (n, d) tensor of sampled points.
        """
        if sample_method != "uniform":
            raise NotImplementedError(f"sample_method '{sample_method}' not supported")

        lb, ub = self.bounds[0], self.bounds[1]
        return torch.rand(n, self.dim, dtype=self.bounds.dtype, device=self.bounds.device) * (ub - lb) + lb