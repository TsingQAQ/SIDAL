#!/usr/bin/env python3
"""Unit tests for model_builder.py."""

import unittest

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel

from sidal.benchmarking.model_builder import (
    build_and_fit_gp,
    get_model_hyperparameters,
)


class TestBuildAndFitGP(unittest.TestCase):
    """Tests for build_and_fit_gp function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2)
        self.train_Y = torch.sin(self.train_X.sum(dim=-1, keepdim=True))
        self.input_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_basic_model_creation(self):
        """Test that a basic model can be created and fitted."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model, SingleTaskGP)
        # Model should be in eval mode after fitting
        self.assertFalse(model.training)

    def test_model_without_fitting(self):
        """Test model creation without fitting."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model, SingleTaskGP)
        # Model should still be in train mode
        self.assertTrue(model.training)

    def test_1d_train_Y_reshaped(self):
        """Test that 1D train_Y is correctly reshaped to 2D."""
        train_Y_1d = self.train_Y.squeeze()
        self.assertEqual(train_Y_1d.dim(), 1)

        model = build_and_fit_gp(
            self.train_X,
            train_Y_1d,
            self.input_bounds,
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model, SingleTaskGP)

    # =========================================================================
    # Input validation tests
    # =========================================================================

    def test_invalid_train_X_dimension(self):
        """Test that invalid train_X dimension raises error."""
        train_X_3d = torch.rand(5, 10, 2)
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                train_X_3d,
                self.train_Y,
                self.input_bounds,
                use_outcome_transform=True,
            )
        self.assertIn("must be 2-dimensional", str(context.exception))

    def test_invalid_train_Y_shape(self):
        """Test that invalid train_Y shape raises error."""
        train_Y_wrong = torch.rand(10, 3)  # Should be (n, 1)
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                train_Y_wrong,
                self.input_bounds,
                use_outcome_transform=True,
            )
        self.assertIn("must have shape (n, 1)", str(context.exception))

    def test_missing_input_bounds(self):
        """Test that missing input_bounds raises error."""
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                None,
                use_outcome_transform=True,
            )
        self.assertIn("input_bounds must be provided", str(context.exception))

    def test_invalid_input_bounds_shape(self):
        """Test that invalid input_bounds shape raises error."""
        invalid_bounds = torch.tensor([0.0, 1.0])  # Should be (2, d)
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                invalid_bounds,
                use_outcome_transform=True,
            )
        self.assertIn("must have shape (2, d)", str(context.exception))

    def test_mismatched_input_bounds_dimension(self):
        """Test that mismatched input_bounds dimension raises error."""
        wrong_dim_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # 3D bounds for 2D input
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                wrong_dim_bounds,
                use_outcome_transform=True,
            )
        self.assertIn("does not match", str(context.exception))

    def test_outcome_transform_must_be_explicit(self):
        """Test that use_outcome_transform=None raises error."""
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                self.input_bounds,
                use_outcome_transform=None,
            )
        self.assertIn("must be explicitly set", str(context.exception))

    # =========================================================================
    # Kernel type tests
    # =========================================================================

    def test_rbf_kernel(self):
        """Test RBF kernel creation."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="rbf",
            fit_model=False,
            use_outcome_transform=True,
        )
        # Should have ScaleKernel wrapping RBFKernel
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertIsInstance(model.covar_module.base_kernel, RBFKernel)

    def test_matern52_kernel(self):
        """Test Matern 5/2 kernel creation."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="matern52",
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
        self.assertEqual(model.covar_module.base_kernel.nu, 2.5)

    def test_matern32_kernel(self):
        """Test Matern 3/2 kernel creation."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="matern32",
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
        self.assertEqual(model.covar_module.base_kernel.nu, 1.5)

    def test_matern12_kernel(self):
        """Test Matern 1/2 kernel creation."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="matern12",
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model.covar_module, ScaleKernel)
        self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
        self.assertEqual(model.covar_module.base_kernel.nu, 0.5)

    def test_matern_alias(self):
        """Test that 'matern' is alias for 'matern52'."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="matern",
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model.covar_module.base_kernel, MaternKernel)
        self.assertEqual(model.covar_module.base_kernel.nu, 2.5)

    def test_invalid_kernel(self):
        """Test that invalid kernel raises error."""
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                self.input_bounds,
                kernel="invalid_kernel",
                use_outcome_transform=True,
            )
        self.assertIn("Unsupported kernel", str(context.exception))

    def test_kernel_has_ard(self):
        """Test that kernel uses ARD (one lengthscale per dimension)."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            kernel="matern52",
            fit_model=False,
            use_outcome_transform=True,
        )
        # ARD kernel should have lengthscale shape (1, d)
        lengthscale = model.covar_module.base_kernel.lengthscale
        self.assertEqual(lengthscale.shape[-1], 2)  # 2D input

    # =========================================================================
    # Transform tests
    # =========================================================================

    def test_input_transform_applied(self):
        """Test that input transform is applied when requested."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_input_transform=True,
            use_outcome_transform=True,
            fit_model=False,
        )
        self.assertIsNotNone(model.input_transform)
        self.assertIsInstance(model.input_transform, Normalize)

    def test_input_transform_not_applied(self):
        """Test that input transform is not applied when not requested."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_input_transform=False,
            use_outcome_transform=True,
            fit_model=False,
        )
        # When no transform is applied, the attribute may not exist
        self.assertFalse(hasattr(model, 'input_transform') and model.input_transform is not None)

    def test_input_transform_uses_provided_bounds(self):
        """Test that input transform uses the provided bounds."""
        custom_bounds = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            custom_bounds,
            use_input_transform=True,
            use_outcome_transform=True,
            fit_model=False,
        )
        # Check that the transform has the correct bounds
        transform_bounds = model.input_transform.bounds
        self.assertTrue(torch.allclose(transform_bounds, custom_bounds))

    def test_outcome_transform_applied(self):
        """Test that outcome transform is applied when requested."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_outcome_transform=True,
            fit_model=False,
        )
        self.assertIsNotNone(model.outcome_transform)
        self.assertIsInstance(model.outcome_transform, Standardize)

    def test_outcome_transform_not_applied(self):
        """Test that outcome transform is not applied when not requested."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_outcome_transform=False,
            fit_model=False,
        )
        # When no transform is applied, the attribute may not exist
        self.assertFalse(hasattr(model, 'outcome_transform') and model.outcome_transform is not None)

    # =========================================================================
    # Hyperparameter setting tests
    # =========================================================================

    def test_set_lengthscale_scalar(self):
        """Test setting lengthscale as scalar."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"lengthscale": 0.5},
            fit_model=False,
            use_outcome_transform=True,
        )
        lengthscale = model.covar_module.base_kernel.lengthscale
        self.assertTrue(torch.allclose(lengthscale, torch.tensor([[0.5, 0.5]])))

    def test_set_lengthscale_ard(self):
        """Test setting lengthscale as ARD vector."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"lengthscale": [0.3, 0.7]},
            fit_model=False,
            use_outcome_transform=True,
        )
        lengthscale = model.covar_module.base_kernel.lengthscale
        self.assertTrue(torch.allclose(lengthscale, torch.tensor([[0.3, 0.7]])))

    def test_set_outputscale(self):
        """Test setting outputscale."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"outputscale": 2.5},
            fit_model=False,
            use_outcome_transform=True,
        )
        outputscale = model.covar_module.outputscale
        self.assertAlmostEqual(outputscale.item(), 2.5, places=5)

    def test_set_noise(self):
        """Test setting noise."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"noise": 1e-4},
            fit_model=False,
            use_outcome_transform=True,
        )
        noise = model.likelihood.noise
        self.assertAlmostEqual(noise.item(), 1e-4, places=8)

    # =========================================================================
    # Freeze parameter tests
    # =========================================================================

    def test_freeze_lengthscale(self):
        """Test that freeze_lengthscale actually freezes the parameter."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"lengthscale": 0.5, "freeze_lengthscale": True},
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertFalse(model.covar_module.base_kernel.raw_lengthscale.requires_grad)

    def test_freeze_lengthscale_false(self):
        """Test that lengthscale is trainable when not frozen."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"lengthscale": 0.5, "freeze_lengthscale": False},
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertTrue(model.covar_module.base_kernel.raw_lengthscale.requires_grad)

    def test_freeze_outputscale(self):
        """Test that freeze_outputscale actually freezes the parameter."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"outputscale": 1.0, "freeze_outputscale": True},
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertFalse(model.covar_module.raw_outputscale.requires_grad)

    def test_freeze_noise(self):
        """Test that freeze_noise actually freezes the parameter."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"noise": 1e-4, "freeze_noise": True},
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertFalse(model.likelihood.noise_covar.raw_noise.requires_grad)

    def test_freeze_noise_removes_prior(self):
        """Test that freezing noise removes the noise prior."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"noise": 1e-4, "freeze_noise": True},
            use_default_kernel_priors=True,
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertFalse(hasattr(model.likelihood.noise_covar, 'noise_prior'))

    def test_all_frozen_skips_fitting(self):
        """Test that fitting is skipped when all parameters are frozen."""
        # This should not raise any errors even though we're "fitting"
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={
                "lengthscale": 0.5,
                "outputscale": 1.0,
                "noise": 1e-4,
                "freeze_lengthscale": True,
                "freeze_outputscale": True,
                "freeze_noise": True,
            },
            fit_model=True,
            use_outcome_transform=True,
        )
        # Verify all parameters are frozen
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 0)

    def test_partial_freeze_trains_unfrozen(self):
        """Test that only unfrozen parameters are trained."""
        initial_lengthscale = 0.5
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={
                "lengthscale": initial_lengthscale,
                "noise": 1e-4,
                "freeze_lengthscale": True,  # Frozen
                "freeze_noise": True,  # Frozen
                # outputscale NOT frozen - should be trained
            },
            fit_model=True,
            use_outcome_transform=True,
        )
        # Lengthscale should remain at initial value
        lengthscale = model.covar_module.base_kernel.lengthscale
        self.assertTrue(torch.allclose(lengthscale, torch.tensor([[initial_lengthscale, initial_lengthscale]])))

    # =========================================================================
    # use_default_kernel_priors tests
    # =========================================================================

    def test_default_priors_with_outcome_transform_no_scalekernel(self):
        """Test that with priors + outcome_transform, no ScaleKernel is used."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_default_kernel_priors=True,
            use_outcome_transform=True,
            fit_model=False,
        )
        # Should NOT have ScaleKernel (matches SingleTaskGP behavior)
        self.assertNotIsInstance(model.covar_module, ScaleKernel)
        # Should be a base kernel directly (Matern or RBF)
        self.assertTrue(
            isinstance(model.covar_module, MaternKernel) or
            isinstance(model.covar_module, RBFKernel)
        )

    def test_default_priors_without_outcome_transform_has_scalekernel(self):
        """Test that with priors but no outcome_transform, ScaleKernel is used."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_default_kernel_priors=True,
            use_outcome_transform=False,
            fit_model=False,
        )
        # Should have ScaleKernel for outputscale support
        self.assertIsInstance(model.covar_module, ScaleKernel)

    def test_default_priors_invalid_kernel(self):
        """Test that use_default_kernel_priors with unsupported kernel raises error."""
        with self.assertRaises(ValueError) as context:
            build_and_fit_gp(
                self.train_X,
                self.train_Y,
                self.input_bounds,
                kernel="matern32",  # Not supported with priors
                use_default_kernel_priors=True,
                use_outcome_transform=True,
            )
        self.assertIn("only 'rbf' and 'matern52'", str(context.exception))

    def test_default_priors_with_freeze_interaction(self):
        """Test that priors work correctly with freeze flags."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_default_kernel_priors=True,
            hyperparameters={
                "noise": 1e-4,
                "freeze_noise": True,
            },
            fit_model=False,
            use_outcome_transform=True,
        )
        # Noise should be frozen
        self.assertFalse(model.likelihood.noise_covar.raw_noise.requires_grad)
        # Lengthscale should still be trainable
        self.assertTrue(model.covar_module.raw_lengthscale.requires_grad)

    # =========================================================================
    # Mean constant tests
    # =========================================================================

    def test_set_mean_constant(self):
        """Test setting mean constant."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"mean_constant": 0.5},
            fit_model=False,
            use_outcome_transform=True,
        )
        mean_constant = model.mean_module.constant
        self.assertAlmostEqual(mean_constant.item(), 0.5, places=5)

    def test_mean_constant_frozen_by_default(self):
        """Test that mean constant is frozen by default when set."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"mean_constant": 0.5},
            fit_model=False,
            use_outcome_transform=True,
        )
        for param in model.mean_module.parameters():
            self.assertFalse(param.requires_grad)

    # =========================================================================
    # get_model_hyperparameters tests
    # =========================================================================

    def test_get_hyperparameters_basic(self):
        """Test extracting hyperparameters from a model."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={
                "lengthscale": 0.3,
                "outputscale": 2.0,
                "noise": 1e-4,
            },
            fit_model=False,
            use_outcome_transform=True,
        )
        hyperparams = get_model_hyperparameters(model)

        self.assertIn("lengthscale", hyperparams)
        self.assertIn("outputscale", hyperparams)
        self.assertIn("noise", hyperparams)
        self.assertIn("kernel_type", hyperparams)

    def test_get_hyperparameters_frozen_flags(self):
        """Test that frozen flags are correctly reported."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={
                "lengthscale": 0.3,
                "freeze_lengthscale": True,
                "noise": 1e-4,
                "freeze_noise": False,
            },
            fit_model=False,
            use_outcome_transform=True,
        )
        hyperparams = get_model_hyperparameters(model)

        self.assertTrue(hyperparams["lengthscale_frozen"])
        self.assertFalse(hyperparams["noise_frozen"])

    def test_get_hyperparameters_mean_constant(self):
        """Test that mean constant is extracted."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters={"mean_constant": 1.5},
            fit_model=False,
            use_outcome_transform=True,
        )
        hyperparams = get_model_hyperparameters(model)

        self.assertIn("mean_constant", hyperparams)
        self.assertAlmostEqual(hyperparams["mean_constant"], 1.5, places=5)

    # =========================================================================
    # Integration tests
    # =========================================================================

    def test_model_makes_predictions(self):
        """Test that the fitted model can make predictions."""
        model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            use_outcome_transform=True,
        )
        test_X = torch.rand(5, 2)
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(test_X)
            mean = posterior.mean
            variance = posterior.variance

        self.assertEqual(mean.shape, (5, 1))
        self.assertEqual(variance.shape, (5, 1))
        self.assertTrue(torch.all(variance > 0))

    def test_model_predictions_with_transforms(self):
        """Test predictions with both input and outcome transforms."""
        # Use data with non-unit range
        train_X = torch.rand(10, 2) * 10  # [0, 10]
        train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) * 100  # Large scale
        input_bounds = torch.tensor([[0.0, 0.0], [10.0, 10.0]])

        model = build_and_fit_gp(
            train_X,
            train_Y,
            input_bounds,
            use_input_transform=True,
            use_outcome_transform=True,
        )

        test_X = torch.rand(5, 2) * 10
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(test_X)
            mean = posterior.mean

        # Predictions should be in original Y scale (not standardized)
        self.assertTrue(mean.abs().max() > 1)  # Should not be standardized

    def test_reproducibility_with_fixed_hyperparameters(self):
        """Test that fixed hyperparameters give reproducible results."""
        hyperparams = {
            "lengthscale": [0.3, 0.5],
            "outputscale": 1.5,
            "noise": 1e-4,
            "freeze_lengthscale": True,
            "freeze_outputscale": True,
            "freeze_noise": True,
        }

        model1 = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters=hyperparams,
            use_outcome_transform=True,
        )
        model2 = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.input_bounds,
            hyperparameters=hyperparams,
            use_outcome_transform=True,
        )

        test_X = torch.rand(5, 2)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            mean1 = model1.posterior(test_X).mean
            mean2 = model2.posterior(test_X).mean

        self.assertTrue(torch.allclose(mean1, mean2))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_single_data_point(self):
        """Test with a single training point."""
        train_X = torch.tensor([[0.5, 0.5]])
        train_Y = torch.tensor([[1.0]])
        input_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        # Should work but fitting might have issues - just test creation
        model = build_and_fit_gp(
            train_X,
            train_Y,
            input_bounds,
            fit_model=False,
            use_outcome_transform=True,
        )
        self.assertIsInstance(model, SingleTaskGP)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        dim = 10
        train_X = torch.rand(20, dim)
        train_Y = torch.rand(20, 1)
        input_bounds = torch.zeros(2, dim)
        input_bounds[1] = 1.0

        model = build_and_fit_gp(
            train_X,
            train_Y,
            input_bounds,
            fit_model=False,
            use_outcome_transform=True,
        )
        # Check ARD lengthscales
        lengthscale = model.covar_module.base_kernel.lengthscale
        self.assertEqual(lengthscale.shape[-1], dim)

    def test_constant_output(self):
        """Test with constant output values."""
        train_X = torch.rand(10, 2)
        train_Y = torch.ones(10, 1) * 5.0  # Constant
        input_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        # With outcome transform, this should still work
        model = build_and_fit_gp(
            train_X,
            train_Y,
            input_bounds,
            use_outcome_transform=True,
            fit_model=False,  # Fitting might have numerical issues
        )
        self.assertIsInstance(model, SingleTaskGP)


if __name__ == "__main__":
    unittest.main()
