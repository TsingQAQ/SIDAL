#!/usr/bin/env python3
r"""Tests for custom synthetic test functions."""

import math
from itertools import product

import torch
from botorch.utils.testing import (
    BaseTestProblemTestCaseMixIn,
    BotorchTestCase,
    SyntheticTestFunctionTestCaseMixin,
)

from sidal.test_functions import (
    Forrester,
    Gramacy1D,
    Higdon,
)


class TestForrester(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    """Test cases for the Forrester test function."""

    functions = [Forrester(), Forrester(negate=True), Forrester(noise_std=0.1)]

    def test_known_values(self):
        """Test Forrester function at known points."""
        f = Forrester()
        # f(x) = (6x - 2)^2 * sin(12x - 4)
        # At x=0: (6*0 - 2)^2 * sin(12*0 - 4) = 4 * sin(-4)
        x0 = torch.tensor([[0.0]])
        expected_0 = 4.0 * math.sin(-4.0)
        self.assertAlmostEqual(f(x0).item(), expected_0, places=5)

        # At x=1: (6*1 - 2)^2 * sin(12*1 - 4) = 16 * sin(8)
        x1 = torch.tensor([[1.0]])
        expected_1 = 16.0 * math.sin(8.0)
        self.assertAlmostEqual(f(x1).item(), expected_1, places=5)

    def test_bounds(self):
        """Test that bounds are correctly set."""
        f = Forrester()
        self.assertEqual(f.dim, 1)
        self.assertAllClose(f.bounds, torch.tensor([[0.0], [1.0]], dtype=torch.double))

    def test_optimal_at_optimizer(self):
        """Test that optimal value is achieved at the optimizer."""
        f = Forrester()
        x_opt = torch.tensor(f._optimizers)
        f_opt = f(x_opt)
        self.assertAlmostEqual(f_opt.item(), f._optimal_value, places=5)


class TestGramacy1D(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    """Test cases for the Gramacy1D test function."""

    functions = [Gramacy1D(), Gramacy1D(negate=True), Gramacy1D(noise_std=0.1)]

    def test_known_values(self):
        """Test Gramacy1D function at known points."""
        f = Gramacy1D()
        # f(x) = sin(10*pi*x) / (2*x) + (x - 1)^4
        # At x=1: sin(10*pi) / 2 + 0 = 0
        x1 = torch.tensor([[1.0]])
        expected_1 = math.sin(10.0 * math.pi * 1.0) / (2.0 * 1.0) + (1.0 - 1.0) ** 4
        self.assertAlmostEqual(f(x1).item(), expected_1, places=5)

        # At x=2: sin(20*pi) / 4 + 1 = 0 + 1 = 1
        x2 = torch.tensor([[2.0]])
        expected_2 = math.sin(10.0 * math.pi * 2.0) / (2.0 * 2.0) + (2.0 - 1.0) ** 4
        self.assertAlmostEqual(f(x2).item(), expected_2, places=5)

    def test_bounds(self):
        """Test that bounds are correctly set."""
        f = Gramacy1D()
        self.assertEqual(f.dim, 1)
        self.assertAllClose(f.bounds, torch.tensor([[0.5], [2.5]], dtype=torch.double))

    def test_optimal_at_optimizer(self):
        """Test that optimal value is achieved at the optimizer."""
        f = Gramacy1D()
        x_opt = torch.tensor(f._optimizers)
        f_opt = f(x_opt)
        self.assertAlmostEqual(f_opt.item(), f._optimal_value, places=5)


class TestHigdon(
    BotorchTestCase, BaseTestProblemTestCaseMixIn, SyntheticTestFunctionTestCaseMixin
):
    """Test cases for the Higdon test function."""

    functions = [Higdon(), Higdon(negate=True), Higdon(noise_std=0.1)]

    def test_known_values(self):
        """Test Higdon function at known points."""
        f = Higdon()
        # f(x) = sin(2*pi*x/10) + 0.2*sin(2*pi*x/2.5)
        # At x=0: sin(0) + 0.2*sin(0) = 0
        x0 = torch.tensor([[0.0]])
        expected_0 = 0.0
        self.assertAlmostEqual(f(x0).item(), expected_0, places=5)

        # At x=5: sin(pi) + 0.2*sin(4*pi) = 0 + 0 = 0
        x5 = torch.tensor([[5.0]])
        expected_5 = math.sin(2.0 * math.pi * 5.0 / 10.0) + 0.2 * math.sin(
            2.0 * math.pi * 5.0 / 2.5
        )
        self.assertAlmostEqual(f(x5).item(), expected_5, places=5)

        # At x=2.5: sin(pi/2) + 0.2*sin(2*pi) = 1 + 0 = 1
        x25 = torch.tensor([[2.5]])
        expected_25 = math.sin(2.0 * math.pi * 2.5 / 10.0) + 0.2 * math.sin(
            2.0 * math.pi * 2.5 / 2.5
        )
        self.assertAlmostEqual(f(x25).item(), expected_25, places=5)

    def test_bounds(self):
        """Test that bounds are correctly set."""
        f = Higdon()
        self.assertEqual(f.dim, 1)
        self.assertAllClose(f.bounds, torch.tensor([[0.0], [10.0]], dtype=torch.double))

    def test_optimal_at_optimizer(self):
        """Test that optimal value is achieved at the optimizer."""
        f = Higdon()
        x_opt = torch.tensor(f._optimizers)
        f_opt = f(x_opt)
        self.assertAlmostEqual(f_opt.item(), f._optimal_value, places=5)


class TestBatchEvaluation(BotorchTestCase):
    """Test batch evaluation for all custom test functions."""

    def test_batch_shapes(self):
        """Test that all functions handle batch shapes correctly."""
        functions = [Forrester(), Gramacy1D(), Higdon()]
        batch_shapes = [torch.Size(), torch.Size([2]), torch.Size([2, 3])]

        for f, batch_shape in product(functions, batch_shapes):
            X = torch.rand(*batch_shape, f.dim, dtype=torch.double)
            # Scale to bounds
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])
            result = f(X)
            self.assertEqual(result.shape, batch_shape)

    def test_dtype_consistency(self):
        """Test that functions preserve dtype."""
        functions = [Forrester(), Gramacy1D(), Higdon()]
        dtypes = [torch.float, torch.double]

        for f, dtype in product(functions, dtypes):
            f = f.to(dtype=dtype)
            X = torch.rand(5, f.dim, dtype=dtype)
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])
            result = f(X)
            self.assertEqual(result.dtype, dtype)


class TestNegation(BotorchTestCase):
    """Test negation functionality for all custom test functions."""

    def test_negate(self):
        """Test that negate=True correctly negates the function."""
        for func_class in [Forrester, Gramacy1D, Higdon]:
            f = func_class()
            f_neg = func_class(negate=True)

            X = torch.rand(10, f.dim, dtype=torch.double)
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])

            result = f(X)
            result_neg = f_neg(X)

            self.assertAllClose(result, -result_neg)


class TestNoise(BotorchTestCase):
    """Test noise functionality for all custom test functions."""

    def test_noise_std(self):
        """Test that noise_std adds noise to the output."""
        torch.manual_seed(42)
        for func_class in [Forrester, Gramacy1D, Higdon]:
            f = func_class()
            f_noisy = func_class(noise_std=0.1)

            X = torch.rand(100, f.dim, dtype=torch.double)
            X = f.bounds[0] + X * (f.bounds[1] - f.bounds[0])

            result_clean = f(X)
            result_noisy = f_noisy(X)

            # Results should differ due to noise
            self.assertFalse(torch.allclose(result_clean, result_noisy))

            # But the true values should be the same
            true_clean = f.evaluate_true(X)
            true_noisy = f_noisy.evaluate_true(X)
            self.assertAllClose(true_clean, true_noisy)
