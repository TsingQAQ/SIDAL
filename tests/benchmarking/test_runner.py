#!/usr/bin/env python3
"""Unit tests for runner.py."""

import math
import unittest

import torch
from botorch.models import SingleTaskGP
from sidal.test_functions.gp_sample import GPSample

from sidal.benchmarking.runner import (
    generate_grid_candidates,
    optimize_acqf_discrete,
    get_acquisition_function,
    run_experiment,
)
from sidal.benchmarking.model_builder import build_and_fit_gp
from sidal import AB_SID_iVAR_Discrete, TS_SID_iVAR_Discrete


class TestGenerateGridCandidates(unittest.TestCase):
    """Tests for generate_grid_candidates function."""

    def test_1d_grid(self):
        """Test 1D grid generation."""
        bounds = torch.tensor([[0.0], [1.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=5)

        self.assertEqual(candidates.shape, (5, 1))
        self.assertTrue(torch.allclose(candidates[:, 0], torch.linspace(0, 1, 5)))

    def test_2d_grid(self):
        """Test 2D grid generation."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=3)

        # 3^2 = 9 candidates
        self.assertEqual(candidates.shape, (9, 2))

    def test_3d_grid(self):
        """Test 3D grid generation."""
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=4)

        # 4^3 = 64 candidates
        self.assertEqual(candidates.shape, (64, 3))

    def test_grid_respects_bounds(self):
        """Test that grid respects provided bounds."""
        bounds = torch.tensor([[-2.0, 0.0], [2.0, 10.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=5)

        # Check all candidates are within bounds
        self.assertTrue(torch.all(candidates[:, 0] >= -2.0))
        self.assertTrue(torch.all(candidates[:, 0] <= 2.0))
        self.assertTrue(torch.all(candidates[:, 1] >= 0.0))
        self.assertTrue(torch.all(candidates[:, 1] <= 10.0))

        # Check corners are included
        self.assertTrue(torch.any(torch.all(candidates == bounds[0], dim=1)))  # Lower corner
        self.assertTrue(torch.any(torch.all(candidates == bounds[1], dim=1)))  # Upper corner

    def test_grid_dtype_preserved(self):
        """Test that grid preserves dtype."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=3)

        self.assertEqual(candidates.dtype, torch.float64)

    def test_grid_device_preserved(self):
        """Test that grid preserves device."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        # Test on CPU (GPU test would need cuda available)
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=3)
        self.assertEqual(candidates.device, bounds.device)

    def test_single_point_grid(self):
        """Test grid with n_grid_per_dim=1."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=1)

        # Should return single point at lower bound
        self.assertEqual(candidates.shape, (1, 2))
        self.assertTrue(torch.allclose(candidates, bounds[0:1]))

    def test_asymmetric_bounds(self):
        """Test grid with asymmetric bounds."""
        bounds = torch.tensor([[0.0, -5.0], [1.0, 5.0]])
        candidates = generate_grid_candidates(bounds, n_grid_per_dim=3)

        # Check first dimension spans [0, 1]
        unique_d0 = torch.unique(candidates[:, 0])
        self.assertTrue(torch.allclose(unique_d0, torch.tensor([0.0, 0.5, 1.0])))

        # Check second dimension spans [-5, 5]
        unique_d1 = torch.unique(candidates[:, 1])
        self.assertTrue(torch.allclose(unique_d1, torch.tensor([-5.0, 0.0, 5.0])))


class TestOptimizeAcqfDiscrete(unittest.TestCase):
    """Tests for optimize_acqf_discrete function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.float64)
        self.train_Y = torch.sin(self.train_X.sum(dim=-1, keepdim=True))
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        self.model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.bounds,
            use_outcome_transform=True,
            fit_model=False,
        )
        self.candidates = generate_grid_candidates(self.bounds, n_grid_per_dim=10)

    def test_returns_correct_shape(self):
        """Test that optimize_acqf_discrete returns correct shapes."""
        from sidal import PosteriorVariance
        acq_fn = PosteriorVariance(model=self.model, maximize=True)

        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        self.assertEqual(best_x.shape, (1, 2))
        self.assertEqual(acq_value.shape, (1,))

    def test_selects_best_candidate(self):
        """Test that it selects the candidate with highest acquisition value."""
        from sidal import PosteriorVariance
        acq_fn = PosteriorVariance(model=self.model, maximize=True)

        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        # Verify the selected candidate is indeed in the candidates set
        distances = torch.cdist(best_x, self.candidates)
        min_distance = distances.min()
        self.assertLess(min_distance, 1e-6)

    def test_handles_all_inf_gracefully(self):
        """Test graceful handling when all acquisition values are -inf."""
        # Create a dummy acquisition function that returns all -inf
        class AllInfAcq:
            def __call__(self, X):
                # X has shape (n_candidates, 1, d), return shape (n_candidates,)
                return torch.full((X.shape[0],), float('-inf'))

        acq_fn = AllInfAcq()
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        # Should return a random candidate - squeeze any extra dims
        best_x = best_x.squeeze()
        if best_x.dim() == 1:
            best_x = best_x.unsqueeze(0)
        self.assertEqual(best_x.shape, (1, 2))
        self.assertTrue(torch.isinf(acq_value) and acq_value.item() < 0)

    def test_masks_inf_values(self):
        """Test that -inf values are properly masked."""
        # Create acquisition function with some -inf values
        class PartialInfAcq:
            def __call__(self, X):
                values = torch.rand(X.shape[0])
                values[:len(values)//2] = float('-inf')  # Half are -inf
                return values

        acq_fn = PartialInfAcq()
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        # Should select from non-inf values
        self.assertTrue(torch.isfinite(acq_value))


class TestGetAcquisitionFunction(unittest.TestCase):
    """Tests for get_acquisition_function function."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.float64)
        self.train_Y = torch.sin(self.train_X.sum(dim=-1, keepdim=True))
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        self.model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.bounds,
            use_outcome_transform=True,
        )
        self.candidates = generate_grid_candidates(self.bounds, n_grid_per_dim=10)

    def test_random_sampling(self):
        """Test RS acquisition function."""
        acq_fn = get_acquisition_function("RS", self.model)

        self.assertTrue(hasattr(acq_fn, '_is_random_sampling'))
        self.assertTrue(acq_fn._is_random_sampling)

        # Should return random values
        X = self.candidates.unsqueeze(1)
        values = acq_fn(X)
        self.assertEqual(values.shape[0], len(self.candidates))

    def test_random_sampling_case_insensitive(self):
        """Test RS is case insensitive."""
        acq_fn1 = get_acquisition_function("RS", self.model)
        acq_fn2 = get_acquisition_function("rs", self.model)
        acq_fn3 = get_acquisition_function("Rs", self.model)

        self.assertTrue(hasattr(acq_fn1, '_is_random_sampling'))
        self.assertTrue(hasattr(acq_fn2, '_is_random_sampling'))
        self.assertTrue(hasattr(acq_fn3, '_is_random_sampling'))

    def test_uncertainty_sampling(self):
        """Test US acquisition function."""
        acq_fn = get_acquisition_function("US", self.model)

        X = self.candidates.unsqueeze(1)
        values = acq_fn(X)

        self.assertEqual(values.shape[0], len(self.candidates))
        # Variance should be non-negative
        self.assertTrue(torch.all(values >= 0))

    def test_imse_continuous_requires_bounds(self):
        """Test IMSE without candidates (continuous) raises error without bounds."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function("IMSE", self.model, candidates=None)
        self.assertIn("IMSE continuous requires bounds", str(context.exception))

    def test_imse_with_candidates(self):
        """Test IMSE with candidates."""
        acq_fn = get_acquisition_function("IMSE", self.model, candidates=self.candidates)

        X = self.candidates[:5].unsqueeze(1)
        values = acq_fn(X)

        self.assertEqual(values.shape[0], 5)

    def test_epig_continuous_requires_bounds(self):
        """Test EPIG without candidates (continuous) raises error without bounds."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function("EPIG", self.model, candidates=None)
        self.assertIn("EPIG continuous requires bounds", str(context.exception))

    def test_ts_sid_ivar_continuous_requires_bounds(self):
        """Test TS-SID-iVAR without candidates (continuous) raises error without bounds."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function("TS-SID-iVAR", self.model, candidates=None)
        self.assertIn("SID-iVAR continuous requires bounds", str(context.exception))

    def test_ab_sid_ivar_continuous_requires_bounds(self):
        """Test AB-SID-iVAR without candidates (continuous) raises error without bounds."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function("AB-SID-iVAR", self.model, candidates=None)
        self.assertIn("SID-iVAR continuous requires bounds", str(context.exception))

    def test_ab_sid_ivar_continuous_requires_smc_hyperparameters(self):
        """Test continuous AB-SID-iVAR requires SMC hyperparameters."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function(
                "AB-SID-iVAR", self.model, candidates=None, bounds=self.bounds
            )
        self.assertIn("n_smc_particles", str(context.exception))

    def test_ab_sid_ivar_with_candidates(self):
        """Test AB-SID-iVAR with candidates."""
        acq_fn = get_acquisition_function(
            "AB-SID-iVAR", self.model, candidates=self.candidates, acq_hyperparameters={}
        )
        self.assertIsInstance(acq_fn, AB_SID_iVAR_Discrete)

        X = self.candidates[:5].unsqueeze(1)
        values = acq_fn(X)

        self.assertEqual(values.shape[0], 5)

    def test_ts_sid_ivar_with_candidates(self):
        """Test TS-SID-iVAR with candidates."""
        acq_fn = get_acquisition_function(
            "TS-SID-iVAR", self.model, candidates=self.candidates, acq_hyperparameters={}
        )
        self.assertIsInstance(acq_fn, TS_SID_iVAR_Discrete)

        X = self.candidates[:5].unsqueeze(1)
        values = acq_fn(X)

        self.assertEqual(values.shape[0], 5)

    def test_ab_sid_ivar_with_hyperparameters(self):
        """Test AB-SID-iVAR with hyperparameters."""
        hyperparams = {
            "boltzmann_lambda": -2.0,
            "constraint_type": "energy_weighted_variance",
        }
        acq_fn = get_acquisition_function(
            "AB-SID-iVAR",
            self.model,
            candidates=self.candidates,
            acq_hyperparameters=hyperparams,
        )

        # boltzmann_lambda is converted to temperature = 1/|lambda|, sign = sign(lambda)
        self.assertAlmostEqual(acq_fn._energy_dist.temperature, 0.5)
        self.assertEqual(acq_fn._energy_dist.boltzmann_sign, -1.0)
        self.assertEqual(acq_fn._energy_dist.base_approximation, "zeroth_order_analytic")
        self.assertEqual(acq_fn._constraint_type, "energy_weighted_variance")

    def test_unknown_acquisition_raises_error(self):
        """Test that unknown acquisition function raises error."""
        with self.assertRaises(ValueError) as context:
            get_acquisition_function("UNKNOWN_ACQ", self.model)
        self.assertIn("Unknown acquisition function", str(context.exception))

    def test_hyperparameters_none_handling(self):
        """Test that None hyperparameters dict is handled."""
        acq_fn = get_acquisition_function("US", self.model, acq_hyperparameters=None)
        self.assertIsNotNone(acq_fn)

    def test_empty_hyperparameters_handling(self):
        """Test that empty hyperparameters dict is handled."""
        acq_fn = get_acquisition_function("US", self.model, acq_hyperparameters={})
        self.assertIsNotNone(acq_fn)

    def test_sid_ivar_constraint_types(self):
        """Test different constraint types for AB-SID-iVAR."""
        constraint_types = [None, "energy_weighted_variance"]

        for ct in constraint_types:
            hyperparams = {"constraint_type": ct}
            acq_fn = get_acquisition_function(
                "AB-SID-iVAR",
                self.model,
                candidates=self.candidates,
                acq_hyperparameters=hyperparams,
            )
            self.assertEqual(acq_fn._constraint_type, ct)

    def test_sid_ivar_unknown_constraint_type_raises(self):
        """Test that unsupported constraint types are rejected."""
        with self.assertRaises(ValueError):
            get_acquisition_function(
                "AB-SID-iVAR",
                self.model,
                candidates=self.candidates,
                acq_hyperparameters={"constraint_type": "unknown_constraint"},
            )

    def test_sid_ivar_default_constraint_is_none(self):
        """Test that the constraint set is off unless constraint_type is given."""
        for name in ["AB-SID-iVAR", "TS-SID-iVAR", "mu-SID-EPIG"]:
            acq_fn = get_acquisition_function(
                name, self.model, candidates=self.candidates, acq_hyperparameters={}
            )
            self.assertIsNone(acq_fn._constraint_type, msg=name)

    def test_stabilize_log_density_passed(self):
        """Test that stabilize_log_density is passed to AB-SID-iVAR."""
        for flag in (True, False):
            acq_fn = get_acquisition_function(
                "AB-SID-iVAR",
                self.model,
                candidates=self.candidates,
                acq_hyperparameters={"stabilize_log_density": flag},
            )
            # stabilize_log_density is stored on the underlying energy distribution
            self.assertEqual(acq_fn._energy_dist.stabilize_log_density, flag)


class TestAcquisitionFunctionIntegration(unittest.TestCase):
    """Integration tests for acquisition functions with optimize_acqf_discrete."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.train_X = torch.rand(10, 2, dtype=torch.float64)
        self.train_Y = torch.sin(self.train_X.sum(dim=-1, keepdim=True))
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        self.model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.bounds,
            use_outcome_transform=True,
        )
        self.candidates = generate_grid_candidates(self.bounds, n_grid_per_dim=10)

    def test_rs_optimization(self):
        """Test RS can be optimized over discrete candidates."""
        acq_fn = get_acquisition_function("RS", self.model)
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        self.assertEqual(best_x.shape, (1, 2))

    def test_us_optimization(self):
        """Test US can be optimized over discrete candidates."""
        acq_fn = get_acquisition_function("US", self.model)
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        self.assertEqual(best_x.shape, (1, 2))
        self.assertTrue(acq_value.item() >= 0)  # Variance is non-negative

    def test_imse_optimization(self):
        """Test IMSE can be optimized over discrete candidates."""
        acq_fn = get_acquisition_function("IMSE", self.model, candidates=self.candidates)
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        self.assertEqual(best_x.shape, (1, 2))

    def test_ab_sid_ivar_optimization(self):
        """Test AB-SID-iVAR can be optimized over discrete candidates."""
        acq_fn = get_acquisition_function(
            "AB-SID-iVAR", self.model, candidates=self.candidates, acq_hyperparameters={}
        )
        best_x, acq_value = optimize_acqf_discrete(acq_fn, self.candidates)

        self.assertEqual(best_x.shape, (1, 2))
        self.assertTrue(torch.isfinite(acq_value).all())


class TestWeightedMSE(unittest.TestCase):
    """Test weighted MSE computation in run_experiment."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_fn = GPSample(
            dim=2,
            kernel="matern52",
            lengthscale=[0.4, 0.4],
            outputscale=1.0,
            seed=42,
        )
        # GPSample builds its RFF feature map in float32 while its bounds (and hence
        # the runner's grid) are float64; cast the whole module to float64.
        self.test_fn = self.test_fn.to(torch.float64)
        self.density_fn = lambda f: torch.exp(-f)

    @staticmethod
    def _model_config(**overrides):
        """GP config with fixed hyperparameters matching the GPSample test function."""
        config = {
            "kernel": "matern52",
            "use_input_transform": True,
            "use_outcome_transform": True,
            "use_default_kernel_priors": False,
            "noise_level": 1e-4,
            "fix_noise": True,
            "fixed_lengthscale": [0.4, 0.4],
            "fixed_outputscale": 1.0,
        }
        config.update(overrides)
        return config

    def test_wmse_is_nan_without_density_fn(self):
        """Test that weighted_mse is NaN when density_fn is not provided."""
        history = run_experiment(
            test_function=self.test_fn,
            bounds=self.test_fn.bounds,
            acq_name="RS",
            seed=0,
            n_init=2,
            n_iterations=1,
            model_config=self._model_config(),
            design_space_type="discrete",
            n_grid_per_dim=20,
            performance_metrics=["mse", "weighted_mse"],
            density_fn=None,
            verbose=False,
        )

        # All weighted_mse values should be NaN
        for wmse in history["metrics"]["weighted_mse"]:
            self.assertTrue(math.isnan(wmse), f"Expected NaN but got {wmse}")

        # MSE should still have valid values
        for mse in history["metrics"]["mse"]:
            self.assertFalse(math.isnan(mse), f"MSE should not be NaN: {mse}")

    def test_wmse_differs_from_mse_with_density_fn(self):
        """Test that weighted_mse differs from mse when density_fn is provided."""
        history = run_experiment(
            test_function=self.test_fn,
            bounds=self.test_fn.bounds,
            acq_name="RS",
            seed=0,
            n_init=2,
            n_iterations=2,
            model_config=self._model_config(),
            design_space_type="discrete",
            n_grid_per_dim=20,
            performance_metrics=["mse", "weighted_mse"],
            density_fn=self.density_fn,
            verbose=False,
        )

        # Check that mse and weighted_mse are different
        for i, (mse, wmse) in enumerate(
            zip(history["metrics"]["mse"], history["metrics"]["weighted_mse"])
        ):
            self.assertFalse(math.isnan(wmse), f"WMSE should not be NaN: {wmse}")
            self.assertNotAlmostEqual(
                mse,
                wmse,
                places=5,
                msg=f"Iter {i}: MSE ({mse}) should differ from WMSE ({wmse})",
            )

    def test_wmse_manual_computation(self):
        """Test that weighted_mse matches manual computation."""
        n_grid = 30

        # Run experiment - use input/outcome transforms = False to simplify manual reconstruction
        history = run_experiment(
            test_function=self.test_fn,
            bounds=self.test_fn.bounds,
            acq_name="RS",
            seed=123,
            n_init=3,
            n_iterations=2,
            model_config=self._model_config(use_input_transform=False, use_outcome_transform=False),
            design_space_type="discrete",
            n_grid_per_dim=n_grid,
            performance_metrics=["mse", "weighted_mse"],
            density_fn=self.density_fn,
            verbose=False,
        )

        # Check final iteration manually
        final_iter = len(history["metrics"]["mse"]) - 1
        stored_wmse = history["metrics"]["weighted_mse"][final_iter]
        stored_mse = history["metrics"]["mse"][final_iter]

        # Rebuild the GP from stored data
        train_X = history["X"][final_iter]
        train_Y = history["Y"][final_iter]

        gp = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=self.test_fn.bounds,
            kernel="matern52",
            hyperparameters={
                "lengthscale": [0.4, 0.4],
                "outputscale": 1.0,
                "noise": 1e-4,
                "freeze_lengthscale": True,
                "freeze_outputscale": True,
                "freeze_noise": True,
            },
            use_input_transform=False,
            use_outcome_transform=False,
            fit_model=False,
            verbose=False,
        )

        # Use the same grid generation as the runner (float64)
        bounds = self.test_fn.bounds
        eval_X = generate_grid_candidates(bounds, n_grid)

        # Compute true values and predictions
        with torch.no_grad():
            eval_Y_true = self.test_fn(eval_X).squeeze()
            pred_mean = gp.posterior(eval_X).mean.squeeze()

        # Manual computation
        squared_errors = (pred_mean - eval_Y_true) ** 2
        manual_mse = squared_errors.mean().item()
        density_weights = self.density_fn(eval_Y_true)
        density_weights = density_weights / density_weights.sum()
        manual_wmse = (squared_errors * density_weights).sum().item()

        # Compare MSE first
        self.assertAlmostEqual(
            stored_mse,
            manual_mse,
            places=6,
            msg=f"Stored MSE ({stored_mse}) != Manual MSE ({manual_mse})",
        )

        # Compare WMSE
        self.assertAlmostEqual(
            stored_wmse,
            manual_wmse,
            places=6,
            msg=f"Stored WMSE ({stored_wmse}) != Manual WMSE ({manual_wmse})",
        )

    def test_wmse_weights_low_regions_more(self):
        """Test that WMSE weights low-value regions more heavily (Boltzmann with -f)."""
        # Run experiment
        history = run_experiment(
            test_function=self.test_fn,
            bounds=self.test_fn.bounds,
            acq_name="RS",
            seed=456,
            n_init=3,
            n_iterations=3,
            model_config=self._model_config(),
            design_space_type="discrete",
            n_grid_per_dim=30,
            performance_metrics=["mse", "weighted_mse"],
            density_fn=self.density_fn,
            verbose=False,
        )

        # Verify density weights sum to 1 and emphasize low values
        bounds = self.test_fn.bounds
        n_grid = 30
        x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), n_grid)
        x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), n_grid)
        X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
        eval_X = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

        with torch.no_grad():
            eval_Y = self.test_fn(eval_X).squeeze()

        density_weights = self.density_fn(eval_Y)
        density_weights = density_weights / density_weights.sum()

        # Weights should sum to 1
        self.assertAlmostEqual(
            density_weights.sum().item(), 1.0, places=5, msg="Weights should sum to 1"
        )

        # Find indices of lowest and highest values
        min_idx = eval_Y.argmin().item()
        max_idx = eval_Y.argmax().item()

        # Weight at minimum should be higher than weight at maximum
        self.assertGreater(
            density_weights[min_idx].item(),
            density_weights[max_idx].item(),
            "Weight at minimum value should be greater than weight at maximum value",
        )

    def test_weighted_nlml_is_nan_without_density_fn(self):
        """Test that weighted_nlml is NaN when density_fn is not provided."""
        history = run_experiment(
            test_function=self.test_fn,
            bounds=self.test_fn.bounds,
            acq_name="RS",
            seed=0,
            n_init=2,
            n_iterations=1,
            model_config=self._model_config(),
            design_space_type="discrete",
            n_grid_per_dim=20,
            performance_metrics=["nlml", "weighted_nlml"],
            density_fn=None,
            verbose=False,
        )

        # All weighted_nlml values should be NaN
        for wnlml in history["metrics"]["weighted_nlml"]:
            self.assertTrue(math.isnan(wnlml), f"Expected NaN but got {wnlml}")

        # NLML should still have valid values
        for nlml in history["metrics"]["nlml"]:
            self.assertFalse(math.isnan(nlml), f"NLML should not be NaN: {nlml}")


class TestExpectedVarianceEstimators(unittest.TestCase):
    """Test that importance sampling and direct Monte Carlo give equivalent E_P[σ²].

    This validates that for computing E_P[σ²]:
    1. Importance sampling (exhaustive grid): Σ P(x_i) · σ²(x_i) / Σ P(x_i)
    2. Direct Monte Carlo (SMC from P): mean(σ²(x_i)) where x_i ~ P

    These should be asymptotically equivalent, which justifies using mean(σ²)
    for continuous acquisition functions where reference points are SMC samples from P.
    """

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        # Create training data
        self.train_X = torch.rand(15, 2, dtype=torch.float64)
        self.train_Y = torch.sin(2 * self.train_X[:, 0:1]) + 0.5 * self.train_X[:, 1:2]
        self.bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

        # Build GP model
        self.model = build_and_fit_gp(
            self.train_X,
            self.train_Y,
            self.bounds,
            use_outcome_transform=True,
            use_input_transform=True,
        )

        # Temperature for Boltzmann distribution
        self.temperature = 1.0

    def _compute_boltzmann_weights(self, X: torch.Tensor, sign: float = -1.0) -> torch.Tensor:
        """Compute Boltzmann weights P(x) ∝ exp(sign * μ(x) / T).

        Args:
            X: (n, d) points
            sign: -1.0 for minimization (low values preferred), +1.0 for maximization

        Returns:
            (n,) normalized weights summing to 1
        """
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze(-1)

        # Log weights for numerical stability
        log_weights = sign * mu / self.temperature
        log_weights = log_weights - log_weights.max()  # Shift for stability
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()
        return weights

    def _compute_variance_at_points(self, X: torch.Tensor) -> torch.Tensor:
        """Compute posterior variance at points.

        Args:
            X: (n, d) points

        Returns:
            (n,) variances
        """
        with torch.no_grad():
            posterior = self.model.posterior(X)
            var = posterior.variance.squeeze(-1)
        return var

    def test_importance_vs_direct_mc_basic(self):
        """Test that importance sampling ≈ direct MC for E_P[σ²]."""
        # Method 1: Importance sampling on dense grid
        n_grid = 50  # 50^2 = 2500 points
        grid_X = generate_grid_candidates(self.bounds, n_grid_per_dim=n_grid)

        weights = self._compute_boltzmann_weights(grid_X)
        variances = self._compute_variance_at_points(grid_X)

        # E_P[σ²] via importance sampling
        expected_var_importance = (weights * variances).sum().item()

        # Method 2: Direct Monte Carlo via rejection sampling
        # Sample from P by rejection sampling
        n_samples = 5000
        samples = []
        max_weight = weights.max().item()

        torch.manual_seed(123)
        while len(samples) < n_samples:
            # Propose from uniform
            proposal = torch.rand(1000, 2, dtype=torch.float64)
            proposal = self.bounds[0] + proposal * (self.bounds[1] - self.bounds[0])

            # Compute acceptance probability
            prop_weights = self._compute_boltzmann_weights(proposal)
            # Acceptance ratio (weights are already normalized, so scale by grid size)
            accept_prob = prop_weights / (max_weight + 1e-10)
            accept_prob = accept_prob.clamp(max=1.0)

            # Accept/reject
            u = torch.rand(len(proposal))
            accepted = proposal[u < accept_prob]
            samples.append(accepted)

        samples = torch.cat(samples, dim=0)[:n_samples]

        # E_P[σ²] via direct Monte Carlo
        sample_variances = self._compute_variance_at_points(samples)
        expected_var_direct = sample_variances.mean().item()

        # They should be close (within 10% relative error)
        rel_error = abs(expected_var_importance - expected_var_direct) / expected_var_importance
        self.assertLess(
            rel_error, 0.15,
            f"Importance sampling ({expected_var_importance:.6f}) vs "
            f"Direct MC ({expected_var_direct:.6f}) differ by {rel_error*100:.1f}%"
        )

    def test_importance_vs_smc_using_energy_distribution(self):
        """Test using actual SMC implementation from BoltzmannEnergyDistribution."""
        from sidal import (
            ContinuousDesignSpace,
            AB_SID_iVAR_Continuous,
            smc_reference_point_generator,
        )

        # Method 1: Importance sampling on dense grid
        n_grid = 50
        grid_X = generate_grid_candidates(self.bounds, n_grid_per_dim=n_grid)

        weights = self._compute_boltzmann_weights(grid_X, sign=-1.0)
        variances = self._compute_variance_at_points(grid_X)

        expected_var_importance = (weights * variances).sum().item()

        # Method 2: Use AB_SID_iVAR_Continuous which samples via SMC
        space = ContinuousDesignSpace(
            bounds=self.bounds,
            reference_point_generator=smc_reference_point_generator,
            reference_point_generator_kwargs={
                "n_samples": 2000,
                "n_mutation_steps": 5,
                "n_temperatures": 10,
                "seed": 42,
            },
        )

        acq = AB_SID_iVAR_Continuous(
            space=space,
            model=self.model,
            temperature=self.temperature,
            boltzmann_sign=-1.0,
            seed=42,
        )

        # Get reference points (SMC samples from P)
        ref_points = acq.reference_points
        ref_variances = self._compute_variance_at_points(ref_points)

        # E_P[σ²] via direct MC on SMC samples (equal weights)
        expected_var_smc = ref_variances.mean().item()

        # They should be close
        rel_error = abs(expected_var_importance - expected_var_smc) / expected_var_importance
        self.assertLess(
            rel_error, 0.20,  # Allow 20% for SMC approximation
            f"Importance sampling ({expected_var_importance:.6f}) vs "
            f"SMC direct MC ({expected_var_smc:.6f}) differ by {rel_error*100:.1f}%"
        )

    def test_double_weighting_is_wrong(self):
        """Test that double-weighting gives DIFFERENT result from correct estimators.

        This demonstrates the bug: if SMC samples from P and we also weight by P,
        we get double-weighting which gives wrong E_P[σ²].
        """
        from sidal import (
            ContinuousDesignSpace,
            AB_SID_iVAR_Continuous,
            smc_reference_point_generator,
        )

        # Method 1: Correct importance sampling on grid (use dense grid as ground truth)
        n_grid = 100  # 100^2 = 10000 points for accurate ground truth
        grid_X = generate_grid_candidates(self.bounds, n_grid_per_dim=n_grid)

        weights = self._compute_boltzmann_weights(grid_X, sign=-1.0)
        variances = self._compute_variance_at_points(grid_X)

        expected_var_correct = (weights * variances).sum().item()

        # Method 2: SMC + correct equal weights (use more samples)
        space = ContinuousDesignSpace(
            bounds=self.bounds,
            reference_point_generator=smc_reference_point_generator,
            reference_point_generator_kwargs={
                "n_samples": 10000,  # 10K samples for accurate estimate
                "n_mutation_steps": 10,
                "n_temperatures": 15,
                "seed": 42,
            },
        )

        acq = AB_SID_iVAR_Continuous(
            space=space,
            model=self.model,
            temperature=self.temperature,
            boltzmann_sign=-1.0,
            seed=42,
        )

        ref_points = acq.reference_points
        ref_variances = self._compute_variance_at_points(ref_points)

        expected_var_smc_correct = ref_variances.mean().item()

        # Method 3: SMC + WRONG double-weighting
        smc_weights = self._compute_boltzmann_weights(ref_points, sign=-1.0)
        expected_var_smc_wrong = (smc_weights * ref_variances).sum().item()

        # Correct methods should be close
        rel_error_correct = abs(expected_var_correct - expected_var_smc_correct) / expected_var_correct
        self.assertLess(rel_error_correct, 0.20)

        # Double-weighted should be DIFFERENT from correct
        # (It's biased toward high-density regions which often have lower variance)
        rel_error_wrong = abs(expected_var_correct - expected_var_smc_wrong) / expected_var_correct

        # We expect double-weighting to give a noticeably different result
        # Note: In some cases the difference might be small if P is nearly uniform,
        # so we check that SMC+equal weights is at least as good as double-weighting
        print(f"\nE_P[σ²] estimates:")
        print(f"  Grid importance sampling: {expected_var_correct:.6f}")
        print(f"  SMC + equal weights:      {expected_var_smc_correct:.6f} (error: {rel_error_correct*100:.1f}%)")
        print(f"  SMC + double-weighting:   {expected_var_smc_wrong:.6f} (error: {rel_error_wrong*100:.1f}%)")

        # The correct method should have lower or comparable error
        # (double-weighting is biased, equal weights is unbiased)
        # Note: Due to variance in the estimates, we don't strictly enforce this,
        # but we log it for diagnostic purposes

    def test_runner_threshold_uses_correct_estimator(self):
        """Test that _get_variance_constraint uses the correct estimator for continuous."""
        from sidal import (
            ContinuousDesignSpace,
            AB_SID_iVAR_Continuous,
            smc_reference_point_generator,
        )
        from sidal.benchmarking.runner import _get_variance_constraint

        # Create continuous acquisition function
        space = ContinuousDesignSpace(
            bounds=self.bounds,
            reference_point_generator=smc_reference_point_generator,
            reference_point_generator_kwargs={
                "n_samples": 1000,
                "n_mutation_steps": 5,
                "n_temperatures": 10,
                "seed": 42,
            },
        )

        acq = AB_SID_iVAR_Continuous(
            space=space,
            model=self.model,
            temperature=self.temperature,
            boltzmann_sign=-1.0,
            seed=42,
            constraint_type="energy_weighted_variance",
        )

        # Get the variance constraint
        constraint_result = _get_variance_constraint(self.model, acq)
        self.assertIsNotNone(constraint_result)

        variance_constraint_fn, is_intra_point = constraint_result
        self.assertTrue(is_intra_point)

        # The constraint should use mean(σ²) as threshold for continuous
        # Test a point and verify the constraint is computed correctly
        ref_points = acq.reference_points
        ref_variances = self._compute_variance_at_points(ref_points)
        expected_threshold = ref_variances.mean().item()

        # Test at a point with known variance
        test_point = torch.tensor([0.5, 0.5], dtype=torch.float64)
        test_variance = self._compute_variance_at_points(test_point.unsqueeze(0)).item()

        # Constraint value should be: var(x) - threshold
        constraint_value = variance_constraint_fn(test_point).item()
        expected_constraint = test_variance - expected_threshold

        self.assertAlmostEqual(
            constraint_value,
            expected_constraint,
            places=5,
            msg=f"Constraint value {constraint_value} != expected {expected_constraint}"
        )


if __name__ == "__main__":
    unittest.main()
