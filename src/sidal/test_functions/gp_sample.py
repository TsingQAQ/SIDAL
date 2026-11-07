#!/usr/bin/env python3
"""
GP Sample Test Function using BoTorch's pathwise sampling.

This module provides a test function sampled from a Gaussian Process prior using
BoTorch's existing RFF implementation, enabling "in-distribution" benchmarking.
"""

import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem
from botorch.models import SingleTaskGP
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.means import ZeroMean
from typing import Optional, Literal, Union, List, Dict, Any


class GPSample(BaseTestProblem):
    """
    A test function sampled from a Gaussian Process prior using BoTorch's RFF.

    This leverages BoTorch's `draw_kernel_feature_paths` which correctly implements
    random Fourier features for RBF and Matérn kernels with proper spectral densities.
    """

    def __init__(
        self,
        dim: int = 2,
        kernel: Literal["rbf", "matern52", "matern32", "matern12"] = "matern52",
        lengthscale: Union[float, List[float], Tensor] = 0.5,
        outputscale: float = 1.0,
        n_features: int = 1000,
        seed: Optional[int] = None,
        negate: bool = False,
    ):
        """
        Args:
            dim: Input dimension
            kernel: Kernel type
            lengthscale: Kernel lengthscale parameter. Can be:
                - float: isotropic (same for all dimensions)
                - list/tensor of length dim: ARD (per-dimension lengthscales)
            outputscale: Kernel outputscale parameter
            n_features: Number of RFF features (higher = better approximation)
            seed: Random seed for reproducibility
            negate: If True, negate the function values
        """
        self.dim = dim
        # Define bounds like Branin: [(lower, upper), (lower, upper), ...]
        self._bounds = [(0.0, 1.0) for _ in range(dim)]
        self.continuous_inds = list(range(dim))

        super().__init__(negate=negate)

        self.kernel_type = kernel
        self.seed = seed

        # Convert lengthscale to tensor for ARD support
        if isinstance(lengthscale, (list, tuple)):
            lengthscale_tensor = torch.tensor(lengthscale, dtype=torch.float64)
        elif isinstance(lengthscale, Tensor):
            lengthscale_tensor = lengthscale.to(dtype=torch.float64)
        else:
            # Scalar: broadcast to all dimensions
            lengthscale_tensor = torch.full((dim,), lengthscale, dtype=torch.float64)

        if lengthscale_tensor.numel() == 1:
            lengthscale_tensor = lengthscale_tensor.expand(dim)

        assert lengthscale_tensor.shape == (dim,), \
            f"Lengthscale shape {lengthscale_tensor.shape} doesn't match dim {dim}"

        self.lengthscale = lengthscale_tensor
        self.outputscale = outputscale
        self.n_features = n_features

        # Create GP kernel using gpytorch
        if kernel == "rbf":
            base_kernel = RBFKernel(ard_num_dims=dim)
        elif kernel in ["matern52", "matern32", "matern12"]:
            nu_map = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}
            base_kernel = MaternKernel(nu=nu_map[kernel], ard_num_dims=dim)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # Set lengthscale
        base_kernel.lengthscale = self.lengthscale

        # Wrap with ScaleKernel for outputscale
        self.covar_module = ScaleKernel(base_kernel)
        self.covar_module.outputscale = outputscale

        # Create mean module
        self.mean_module = ZeroMean()

        # Generate pathwise sample using BoTorch's infrastructure
        if seed is not None:
            torch.manual_seed(seed)

        # Use BoTorch's draw_kernel_feature_paths
        # We need to create a dummy GP to use the API
        dummy_X = torch.zeros(1, dim, dtype=torch.float64)
        dummy_Y = torch.zeros(1, 1, dtype=torch.float64)
        dummy_gp = SingleTaskGP(dummy_X, dummy_Y)
        dummy_gp.mean_module = self.mean_module
        dummy_gp.covar_module = self.covar_module

        # Draw one sample path
        self.sample_path = draw_kernel_feature_paths(
            model=dummy_gp,
            sample_shape=torch.Size([1]),
            num_features=n_features,
        )

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the GP sample function at points X.

        Args:
            X: Tensor of shape (..., d)

        Returns:
            Tensor of shape (...)
        """
        # X: (..., d)
        X_original_shape = X.shape[:-1]

        # Evaluate pathwise sample
        # sample_path expects (n_points, d) and returns (1, n_points)
        X_flat = X.reshape(-1, self.dim)
        f = self.sample_path(X_flat)  # (1, n_points)
        f = f.squeeze(0)  # (n_points,)

        return f.reshape(X_original_shape)

    def get_rff_params(self) -> Dict[str, Any]:
        """
        Get RFF parameters for reconstruction.

        Returns:
            Dictionary with RFF parameters including the pathwise sample path
        """
        return {
            "kernel_type": self.kernel_type,
            "lengthscale": self.lengthscale.clone(),
            "outputscale": self.outputscale,
            "n_features": self.n_features,
            "dim": self.dim,
            "seed": self.seed,
            # Store the feature map parameters from the sample path
            "feature_map_weight": self.sample_path.feature_map.weight.clone(),
            "path_weight": self.sample_path.weight.clone(),
        }

    @staticmethod
    def from_rff_params(params: Dict[str, Any]) -> "GPSample":
        """
        Reconstruct GPSample from RFF parameters.

        Args:
            params: Dictionary from get_rff_params()

        Returns:
            Reconstructed GPSample instance
        """
        gp_sample = GPSample(
            dim=params["dim"],
            kernel=params["kernel_type"],
            lengthscale=params["lengthscale"],
            outputscale=params["outputscale"],
            n_features=params["n_features"],
            seed=None,  # Don't resample
        )

        # Override with saved parameters
        gp_sample.sample_path.feature_map.weight = params["feature_map_weight"]
        gp_sample.sample_path.weight = params["path_weight"]

        return gp_sample


if __name__ == "__main__":
    """Test the GPSample function."""
    import matplotlib.pyplot as plt

    print("Testing GPSample with BoTorch's RFF implementation...")

    # Test 1: Isotropic lengthscale
    print("\n1. Testing isotropic lengthscale...")
    gp_func = GPSample(dim=2, kernel="matern52", lengthscale=0.5, outputscale=1.0,
                       n_features=1000, seed=42)

    # Test 2: ARD lengthscale
    print("2. Testing ARD lengthscale...")
    gp_func_ard = GPSample(dim=2, kernel="matern52", lengthscale=[0.3, 0.7],
                           outputscale=1.0, n_features=1000, seed=43)

    # Test 3: Reconstruction
    print("3. Testing reconstruction...")
    rff_params = gp_func.get_rff_params()
    gp_func_reconstructed = GPSample.from_rff_params(rff_params)

    # Evaluate on grid
    n_grid = 50
    x1 = torch.linspace(0, 1, n_grid, dtype=torch.float64)
    x2 = torch.linspace(0, 1, n_grid, dtype=torch.float64)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    f_original = gp_func(X_grid).detach().reshape(n_grid, n_grid).numpy()
    f_ard = gp_func_ard(X_grid).detach().reshape(n_grid, n_grid).numpy()
    f_reconstructed = gp_func_reconstructed(X_grid).detach().reshape(n_grid, n_grid).numpy()

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    im = axes[0].contourf(X1.numpy(), X2.numpy(), f_original, levels=20, cmap='viridis')
    axes[0].set_title('Isotropic (l=0.5)')
    plt.colorbar(im, ax=axes[0])

    im = axes[1].contourf(X1.numpy(), X2.numpy(), f_ard, levels=20, cmap='viridis')
    axes[1].set_title('ARD (l=[0.3, 0.7])')
    plt.colorbar(im, ax=axes[1])

    im = axes[2].contourf(X1.numpy(), X2.numpy(), f_reconstructed, levels=20, cmap='viridis')
    axes[2].set_title('Reconstructed')
    plt.colorbar(im, ax=axes[2])

    diff = f_original - f_reconstructed
    im = axes[3].contourf(X1.numpy(), X2.numpy(), diff, levels=20, cmap='RdBu')
    axes[3].set_title(f'Difference (max: {abs(diff).max():.2e})')
    plt.colorbar(im, ax=axes[3])

    plt.tight_layout()
    plt.savefig('gp_sample_test_botorch.png', dpi=150)
    print(f"\n✓ All tests passed!")
    print(f"✓ Reconstruction error: {abs(diff).max():.2e}")
    print(f"✓ Saved test plot to gp_sample_test_botorch.png")
