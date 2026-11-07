"""Tests for active_learning module."""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


# =============================================================================
# FantasyVarianceMixin tests
# =============================================================================

def test_compute_fantasy_variance_correctness():
    """Test that _compute_fantasy_variance produces correct values."""
    from sidal import (
        qNegIntegratedPosteriorVariance,
        DiscreteDesignSpace,
    )

    torch.manual_seed(42)

    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(30, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=mc_points)
    acq = qNegIntegratedPosteriorVariance(model=model, space=space)

    X_test = torch.rand(5, 2, dtype=torch.double)
    cache = acq._prepare_fantasy_variance_cache(apply_output_transform=True)
    var_after = acq._compute_fantasy_variance(X_test, cache, apply_output_transform=True)

    # Verify shape
    assert var_after.shape == (5, 30), f"Expected (5, 30), got {var_after.shape}"
    # Variance must be positive
    assert (var_after > 0).all(), "Fantasy variance should be positive"
    # Fantasy variance <= prior variance (observing reduces uncertainty)
    var_before = cache.var_before
    assert (var_after < var_before.unsqueeze(0) + 1e-6).all()

    print("✓ _compute_fantasy_variance correctness test passed")


def test_compute_fantasy_variance_with_transforms():
    """Test that mixin handles input/output transforms correctly."""
    from sidal import (
        qNegIntegratedPosteriorVariance,
        DiscreteDesignSpace,
    )

    torch.manual_seed(123)

    train_X = torch.rand(12, 2, dtype=torch.double) * 10
    train_Y = torch.sin(train_X[:, 0:1]) * 100

    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(25, 2, dtype=torch.double) * 10
    space = DiscreteDesignSpace(candidates=mc_points)
    acq = qNegIntegratedPosteriorVariance(model=model, space=space)

    X_test = torch.rand(4, 2, dtype=torch.double) * 10

    cache_with = acq._prepare_fantasy_variance_cache(apply_output_transform=True)
    var_with = acq._compute_fantasy_variance(X_test, cache_with, apply_output_transform=True)

    cache_without = acq._prepare_fantasy_variance_cache(apply_output_transform=False)
    var_without = acq._compute_fantasy_variance(X_test, cache_without, apply_output_transform=False)

    scale_ratio = var_with.mean() / var_without.mean()
    expected_ratio = model.outcome_transform.stdvs.item() ** 2
    assert abs(scale_ratio - expected_ratio) / expected_ratio < 0.01

    print("✓ _compute_fantasy_variance with transforms test passed")


def test_compute_fantasy_variance_batched_consistency():
    """Test that batched computation matches single-point computation."""
    from sidal import (
        qNegIntegratedPosteriorVariance,
        DiscreteDesignSpace,
    )

    torch.manual_seed(456)

    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = (train_X ** 2).sum(dim=-1, keepdim=True)

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(20, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=mc_points)
    acq = qNegIntegratedPosteriorVariance(model=model, space=space)

    X_batch = torch.rand(6, 2, dtype=torch.double)
    cache = acq._prepare_fantasy_variance_cache(apply_output_transform=True)

    var_batched = acq._compute_fantasy_variance(X_batch, cache, apply_output_transform=True)

    var_singles = []
    for i in range(X_batch.shape[0]):
        var_i = acq._compute_fantasy_variance(X_batch[i:i+1], cache, apply_output_transform=True)
        var_singles.append(var_i)
    var_single = torch.cat(var_singles, dim=0)

    torch.testing.assert_close(var_batched, var_single, rtol=1e-10, atol=1e-12)

    print("✓ Batched consistency test passed")


# =============================================================================
# qNegIntegratedPosteriorVariance equivalence tests (vs BoTorch original)
# =============================================================================

def test_qNIPV_equivalence():
    """Test that optimized qNIPV produces same values as botorch's version."""
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as BotorchNIPV,
    )
    from sidal import (
        qNegIntegratedPosteriorVariance as OptimizedNIPV,
        DiscreteDesignSpace,
    )

    torch.manual_seed(42)

    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(50, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=mc_points)

    botorch_acq = BotorchNIPV(model=model, mc_points=mc_points)
    optimized_acq = OptimizedNIPV(model=model, space=space)

    test_X = torch.rand(5, 1, 2, dtype=torch.double)

    with torch.no_grad():
        botorch_values = botorch_acq(test_X)
        optimized_values = optimized_acq(test_X)

    torch.testing.assert_close(botorch_values, optimized_values, rtol=1e-4, atol=1e-6)

    print(f"✓ qNIPV equivalence test passed")
    print(f"  Max abs diff: {(botorch_values - optimized_values).abs().max().item():.2e}")


def test_qNIPV_batch_equivalence():
    """Test equivalence with batch inputs."""
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as BotorchNIPV,
    )
    from sidal import (
        qNegIntegratedPosteriorVariance as OptimizedNIPV,
        DiscreteDesignSpace,
    )

    torch.manual_seed(123)

    train_X = torch.rand(8, 3, dtype=torch.double)
    train_Y = (train_X ** 2).sum(dim=-1, keepdim=True)

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(30, 3, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=mc_points)

    botorch_acq = BotorchNIPV(model=model, mc_points=mc_points)
    optimized_acq = OptimizedNIPV(model=model, space=space)

    test_X = torch.rand(4, 3, 1, 3, dtype=torch.double)  # batch_shape=(4,3), q=1

    with torch.no_grad():
        botorch_values = botorch_acq(test_X)
        optimized_values = optimized_acq(test_X)

    torch.testing.assert_close(botorch_values, optimized_values, rtol=1e-4, atol=1e-6)

    print(f"✓ qNIPV batch equivalence test passed")
    print(f"  Output shape: {botorch_values.shape}")
    print(f"  Max abs diff: {(botorch_values - optimized_values).abs().max().item():.2e}")


def test_qNIPV_with_transforms():
    """Test equivalence when model has input/outcome transforms."""
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as BotorchNIPV,
    )
    from sidal import (
        qNegIntegratedPosteriorVariance as OptimizedNIPV,
        DiscreteDesignSpace,
    )

    torch.manual_seed(456)

    train_X = torch.rand(12, 2, dtype=torch.double) * 10
    train_Y = torch.sin(train_X[:, 0:1]) * 100

    model = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(40, 2, dtype=torch.double) * 10
    space = DiscreteDesignSpace(candidates=mc_points)

    botorch_acq = BotorchNIPV(model=model, mc_points=mc_points)
    optimized_acq = OptimizedNIPV(model=model, space=space)

    test_X = torch.rand(6, 1, 2, dtype=torch.double) * 10

    with torch.no_grad():
        botorch_values = botorch_acq(test_X)
        optimized_values = optimized_acq(test_X)

    torch.testing.assert_close(botorch_values, optimized_values, rtol=1e-3, atol=1e-5)

    print(f"✓ qNIPV with transforms test passed")
    print(f"  Max abs diff: {(botorch_values - optimized_values).abs().max().item():.2e}")


# =============================================================================
# VarianceConstraintMixin tests
# =============================================================================

def test_variance_constraint_mixin_empty_feasible_raises():
    """Test that VarianceConstraintMixin raises when no feasible points exist."""
    from sidal.active_learning import VarianceConstraintMixin

    # Create a simple test class using the mixin
    class TestConstraint(VarianceConstraintMixin):
        pass

    obj = TestConstraint()
    obj._init_constraint(constraint_type="energy_weighted_variance")

    # All variances are the same; unnormalised weights summing to 1.1 push the
    # threshold (Σ w·σ²) above every σ², leaving no feasible point
    sigma_sq = torch.ones(10, dtype=torch.double)
    weights = torch.full((10,), 0.11, dtype=torch.double)

    try:
        obj._compute_constraint_mask(sigma_sq, weight_distribution=weights)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "no feasible points" in str(e)
        print("✓ VarianceConstraintMixin empty feasible set raises correctly")


def test_variance_constraint_mixin_valid():
    """Test that VarianceConstraintMixin works with valid inputs."""
    from sidal.active_learning import VarianceConstraintMixin

    class TestConstraint(VarianceConstraintMixin):
        pass

    # energy_weighted_variance without weights falls back to the uniform mean
    obj = TestConstraint()
    obj._init_constraint(constraint_type="energy_weighted_variance")
    sigma_sq = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.0], dtype=torch.double)
    mask = obj._compute_constraint_mask(sigma_sq)
    # mean = 0.54, so points with sigma_sq >= 0.54 are feasible
    expected = torch.tensor([False, False, False, True, True])
    assert (mask == expected).all(), f"Expected {expected}, got {mask}"
    torch.testing.assert_close(obj._cached_variance_threshold, sigma_sq.mean())

    # No constraint
    obj_none = TestConstraint()
    obj_none._init_constraint(constraint_type=None)
    assert obj_none._constraint_type is None

    # Unknown constraint types are rejected
    obj_bad = TestConstraint()
    try:
        obj_bad._init_constraint(constraint_type="unknown_constraint")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "constraint_type must be one of" in str(e)

    print("✓ VarianceConstraintMixin valid inputs test passed")


def test_variance_constraint_mixin_energy_weighted():
    """Test energy_weighted_variance constraint type."""
    from sidal.active_learning import VarianceConstraintMixin

    class TestConstraint(VarianceConstraintMixin):
        pass

    obj = TestConstraint()
    obj._init_constraint(constraint_type="energy_weighted_variance")

    sigma_sq = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.0], dtype=torch.double)
    # Weights that emphasize high-variance points
    weights = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.4], dtype=torch.double)

    mask = obj._compute_constraint_mask(sigma_sq, weight_distribution=weights)

    # weighted_mean = sum(w * sigma_sq) = 0.05*0.1 + 0.1*0.3 + 0.15*0.5 + 0.3*0.8 + 0.4*1.0
    #              = 0.005 + 0.03 + 0.075 + 0.24 + 0.4 = 0.75
    # So points with sigma_sq >= 0.75 should be in constraint set
    expected = torch.tensor([False, False, False, True, True])
    assert (mask == expected).all(), f"Expected {expected}, got {mask}"

    print("✓ VarianceConstraintMixin energy_weighted_variance test passed")


def test_variance_constraint_mixin_apply_constraint():
    """Test _apply_constraint_to_acquisition method."""
    from sidal import (
        AB_SID_iVAR_Discrete,
        DiscreteDesignSpace,
    )

    torch.manual_seed(42)
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Create discrete space
    candidates = torch.rand(20, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=candidates)

    # Create acquisition with constraint (AB_SID_iVAR_Discrete creates its own BoltzmannEnergyDistribution)
    acq = AB_SID_iVAR_Discrete(
        space=space,
        model=model,
        constraint_type="energy_weighted_variance",
    )

    # Get constraint mask
    mask = acq.constraint_set_mask_at_reference_points
    assert mask.shape == (20,), f"Expected shape (20,), got {mask.shape}"
    assert mask.dtype == torch.bool

    # Test _apply_constraint_to_acquisition
    # Pick some points that are in the constraint set and some that are not
    in_constraint_idx = mask.nonzero().squeeze()[:3]
    out_constraint_idx = (~mask).nonzero().squeeze()[:2]

    if in_constraint_idx.numel() > 0 and out_constraint_idx.numel() > 0:
        X_in = candidates[in_constraint_idx]
        X_out = candidates[out_constraint_idx]
        X_test = torch.cat([X_in, X_out], dim=0)

        acq_values = torch.ones(X_test.shape[0], dtype=torch.double)
        result = acq._apply_constraint_to_acquisition(X_test, acq_values)

        # Points in constraint set should keep their values
        assert (result[:len(X_in)] == 1.0).all(), "In-constraint points should keep value"
        # Points outside should be -inf
        assert (result[len(X_in):] == float('-inf')).all(), "Out-constraint points should be -inf"

    print("✓ VarianceConstraintMixin _apply_constraint_to_acquisition test passed")


def test_boltzmann_energy_distribution_methods():
    """Test BoltzmannEnergyDistribution renamed methods."""
    from sidal.active_learning import BoltzmannEnergyDistribution

    torch.manual_seed(42)
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)

    integration_points = torch.rand(15, 2, dtype=torch.double)

    dist = BoltzmannEnergyDistribution(
        model=model,
        reference_points=integration_points,
        base_approximation="zeroth_order_analytic",
    )

    # Test compute_integration_weights
    weights = dist.compute_integration_weights()
    assert weights.shape == (15,), f"Expected shape (15,), got {weights.shape}"
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6)
    assert (weights >= 0).all(), "Weights should be non-negative"

    # Test get_log_expected_boltzmann_at_reference_points
    log_E = dist.get_log_expected_boltzmann_at_reference_points()
    assert log_E.shape == (15,), f"Expected shape (15,), got {log_E.shape}"
    assert not torch.isnan(log_E).any(), "Log values should not be NaN"

    # Test get_expected_boltzmann_at_reference_points
    E = dist.get_expected_boltzmann_at_reference_points()
    assert E.shape == (15,), f"Expected shape (15,), got {E.shape}"
    assert (E > 0).all(), "Expected Boltzmann values should be positive"
    # Check consistency: E = exp(log_E)
    torch.testing.assert_close(E, torch.exp(log_E), rtol=1e-5, atol=1e-8)

    # Test posterior_stats_at_reference_points (returns dict)
    stats = dist.posterior_stats_at_reference_points
    assert 'mu' in stats and 'sigma' in stats, "Stats should contain mu and sigma"
    assert stats['mu'].shape == (15,), f"Expected mu shape (15,), got {stats['mu'].shape}"
    assert stats['sigma'].shape == (15,), f"Expected sigma shape (15,), got {stats['sigma'].shape}"
    assert (stats['sigma'] > 0).all(), "Sigma should be positive"

    print("✓ BoltzmannEnergyDistribution methods test passed")


def test_ab_sid_ivar_discrete_forward():
    """Test AB_SID_iVAR_Discrete forward pass."""
    from sidal import (
        AB_SID_iVAR_Discrete,
        DiscreteDesignSpace,
    )

    torch.manual_seed(42)
    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    candidates = torch.rand(25, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=candidates)

    # Without constraint
    acq = AB_SID_iVAR_Discrete(
        space=space,
        model=model,
        constraint_type=None,
    )

    # Test forward
    test_X = candidates[:5].unsqueeze(1)  # (5, 1, 2)
    values = acq(test_X)
    assert values.shape == (5,), f"Expected shape (5,), got {values.shape}"

    # With constraint
    acq_constrained = AB_SID_iVAR_Discrete(
        space=space,
        model=model,
        constraint_type="energy_weighted_variance",
    )

    values_constrained = acq_constrained(test_X)
    assert values_constrained.shape == (5,), f"Expected shape (5,), got {values_constrained.shape}"
    # Some values should be -inf (outside constraint set)
    n_inf = (values_constrained == float('-inf')).sum().item()
    n_valid = (values_constrained != float('-inf')).sum().item()
    assert n_inf + n_valid == 5, "All values should be either valid or -inf"

    print("✓ AB_SID_iVAR_Discrete forward test passed")


def test_boltzmann_thompson_with_base_approximation_raises():
    """Test that BoltzmannEnergyDistribution raises when combining Thompson with base_approximation."""
    from sidal.active_learning import BoltzmannEnergyDistribution

    torch.manual_seed(42)
    train_X = torch.rand(5, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)

    integration_points = torch.rand(10, 2, dtype=torch.double)

    # Should raise when combining Thompson with any base_approximation
    for base_approx in ["mean", "zeroth_order_analytic"]:
        try:
            BoltzmannEnergyDistribution(
                model=model,
                reference_points=integration_points,
                base_approximation=base_approx,
                use_thompson_sampling=True,
                stabilize_log_density=False,
            )
            assert False, f"Should have raised ValueError for base_approximation='{base_approx}'"
        except ValueError as e:
            assert "cannot be combined" in str(e)

    # Pure Thompson (base_approximation=None) should work
    dist = BoltzmannEnergyDistribution(
        model=model,
        reference_points=integration_points,
        base_approximation=None,
        use_thompson_sampling=True,
        stabilize_log_density=False,
    )
    P = dist.compute_integration_weights()
    assert P.shape == (10,)
    assert torch.isclose(P.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6)

    print("✓ BoltzmannEnergyDistribution Thompson + base_approximation validation test passed")


def test_boltzmann_stabilize_log_density_with_thompson():
    """Test that stabilize_log_density works with Thompson sampling."""
    from sidal.active_learning import BoltzmannEnergyDistribution

    torch.manual_seed(42)
    train_X = torch.rand(5, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True)) * 1000  # Large scale
    model = SingleTaskGP(train_X, train_Y)

    integration_points = torch.rand(10, 2, dtype=torch.double)

    # Thompson with stabilize_log_density=True should work
    dist_stable = BoltzmannEnergyDistribution(
        model=model,
        reference_points=integration_points,
        base_approximation=None,
        use_thompson_sampling=True,
        stabilize_log_density=True,
        seed=123,
    )
    P_stable = dist_stable.compute_integration_weights()
    assert P_stable.shape == (10,)
    assert torch.isclose(P_stable.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6)

    # Without stabilization should also work (but potentially less stable)
    dist_unstable = BoltzmannEnergyDistribution(
        model=model,
        reference_points=integration_points,
        base_approximation=None,
        use_thompson_sampling=True,
        stabilize_log_density=False,
        seed=123,
    )
    P_unstable = dist_unstable.compute_integration_weights()
    assert torch.isclose(P_unstable.sum(), torch.tensor(1.0, dtype=torch.double), atol=1e-6)

    # Both should give same distribution (just shifted log densities)
    torch.testing.assert_close(P_stable, P_unstable, rtol=1e-5, atol=1e-8)

    # Verify stabilize_log_density raises for unsupported modes
    for base_approx in ["mean", None]:
        try:
            BoltzmannEnergyDistribution(
                model=model,
                reference_points=integration_points,
                base_approximation=base_approx,
                use_thompson_sampling=False,
                stabilize_log_density=True,
            )
            assert False, f"Should have raised ValueError for base_approximation='{base_approx}'"
        except ValueError as e:
            assert "stabilize_log_density=True is only supported" in str(e)

    print("✓ BoltzmannEnergyDistribution stabilize_log_density with Thompson test passed")


# =============================================================================
# Posterior variance optimization tests
# =============================================================================

def test_fast_pred_var_diagonal_variance_exact():
    """
    Test that fast_pred_var + max_eager_kernel_size(512) gives exact diagonal variance.

    This verifies that the optimization in compute_posterior_stats produces
    numerically identical results to the default (dense) computation.
    """
    import gpytorch
    from sidal.active_learning import BoltzmannEnergyDistribution

    torch.manual_seed(42)

    # Create model
    train_X = torch.rand(20, 6, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Test points (1000 particles - typical SMC size)
    X_test = torch.rand(1000, 6, dtype=torch.double)

    # Compute variance with default BoTorch settings (dense computation)
    with torch.no_grad():
        posterior_dense = model.posterior(X_test)
        var_dense = posterior_dense.variance.squeeze(-1)
        mean_dense = posterior_dense.mean.squeeze(-1)

    # Compute variance with optimized settings (fast_pred_var + max_eager_kernel_size)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_eager_kernel_size(512):
        posterior_fast = model.posterior(X_test)
        var_fast = posterior_fast.variance.squeeze(-1)
        mean_fast = posterior_fast.mean.squeeze(-1)

    # Verify mean is exactly the same
    torch.testing.assert_close(mean_dense, mean_fast, rtol=0, atol=1e-14)

    # Verify variance is exactly the same (or within machine precision)
    torch.testing.assert_close(var_dense, var_fast, rtol=0, atol=1e-14)

    print("✓ fast_pred_var diagonal variance exactness test passed")
    print(f"  Max variance diff: {(var_dense - var_fast).abs().max().item():.2e}")
    print(f"  Max mean diff: {(mean_dense - mean_fast).abs().max().item():.2e}")


def test_fast_pred_var_gradient_consistency():
    """
    Test that gradients through fast_pred_var are consistent with dense computation.

    This is critical for MALA sampling which requires accurate gradients.
    """
    import gpytorch

    torch.manual_seed(42)

    # Create model
    train_X = torch.rand(20, 6, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Test points
    X_test = torch.rand(500, 6, dtype=torch.double)

    # Compute gradient with default settings (dense)
    X_dense = X_test.clone().requires_grad_(True)
    posterior_dense = model.posterior(X_dense)
    var_dense = posterior_dense.variance.squeeze(-1)
    loss_dense = var_dense.sum()
    loss_dense.backward()
    grad_dense = X_dense.grad.clone()

    # Compute gradient with optimized settings
    X_fast = X_test.clone().requires_grad_(True)
    with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_eager_kernel_size(512):
        posterior_fast = model.posterior(X_fast)
        var_fast = posterior_fast.variance.squeeze(-1)
        loss_fast = var_fast.sum()
        loss_fast.backward()
    grad_fast = X_fast.grad.clone()

    # Verify gradients match
    torch.testing.assert_close(grad_dense, grad_fast, rtol=1e-5, atol=1e-10)

    print("✓ fast_pred_var gradient consistency test passed")
    print(f"  Max gradient diff: {(grad_dense - grad_fast).abs().max().item():.2e}")
    print(f"  Relative gradient diff: {((grad_dense - grad_fast).abs() / grad_dense.abs().clamp(min=1e-10)).max().item():.2e}")


def test_compute_posterior_stats_uses_optimization():
    """
    Test that BoltzmannEnergyDistribution.compute_posterior_stats uses the optimization.

    Verifies that the optimization is actually applied in the active learning code.
    """
    from sidal.active_learning import BoltzmannEnergyDistribution
    import gpytorch

    torch.manual_seed(42)

    train_X = torch.rand(20, 6, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Create energy distribution with many reference points (like SMC)
    reference_points = torch.rand(2000, 6, dtype=torch.double)

    dist = BoltzmannEnergyDistribution(
        model=model,
        reference_points=reference_points,
        base_approximation="zeroth_order_analytic",
    )

    # Get stats from the optimized method
    mu_opt, sigma_opt = dist.compute_posterior_stats(reference_points)

    # Compare with dense computation
    with torch.no_grad():
        posterior = model.posterior(reference_points)
        mu_dense = posterior.mean.squeeze(-1)
        sigma_dense = posterior.variance.sqrt().squeeze(-1)

    # Should match exactly (or within machine precision)
    torch.testing.assert_close(mu_opt, mu_dense, rtol=0, atol=1e-14)
    torch.testing.assert_close(sigma_opt, sigma_dense, rtol=0, atol=1e-14)

    print("✓ compute_posterior_stats optimization test passed")


# =============================================================================
# qExpectedPredictiveInformationGain tests
# =============================================================================

def test_qEPIG_forward():
    """Test that EPIG forward() works correctly."""
    from sidal import (
        qExpectedPredictiveInformationGain,
        DiscreteDesignSpace,
    )

    torch.manual_seed(789)

    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = torch.sin(train_X.sum(dim=-1, keepdim=True))

    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    mc_points = torch.rand(30, 2, dtype=torch.double)
    space = DiscreteDesignSpace(candidates=mc_points)

    epig = qExpectedPredictiveInformationGain(model=model, space=space)

    # Test with batch_shape
    test_X = torch.rand(4, 3, 1, 2, dtype=torch.double)
    epig_vals = epig(test_X)

    assert epig_vals.shape == (4, 3), f"Expected (4, 3), got {epig_vals.shape}"
    assert (epig_vals > 0).all(), "EPIG should be positive (information gain)"

    print("✓ qEPIG forward test passed")
    print(f"  Output shape: {epig_vals.shape}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing active_learning module")
    print("=" * 60)
    print()

    # Mixin tests
    print("--- FantasyVarianceMixin tests ---")
    test_compute_fantasy_variance_correctness()
    test_compute_fantasy_variance_with_transforms()
    test_compute_fantasy_variance_batched_consistency()
    print()

    # Equivalence tests
    print("--- qNIPV equivalence tests ---")
    test_qNIPV_equivalence()
    test_qNIPV_batch_equivalence()
    test_qNIPV_with_transforms()
    print()

    # VarianceConstraintMixin tests
    print("--- VarianceConstraintMixin tests ---")
    test_variance_constraint_mixin_empty_feasible_raises()
    test_variance_constraint_mixin_valid()
    test_variance_constraint_mixin_energy_weighted()
    test_variance_constraint_mixin_apply_constraint()
    print()

    # BoltzmannEnergyDistribution tests
    print("--- BoltzmannEnergyDistribution tests ---")
    test_boltzmann_energy_distribution_methods()
    test_boltzmann_thompson_with_base_approximation_raises()
    test_boltzmann_stabilize_log_density_with_thompson()
    print()

    # ET_SID tests
    print("--- SID-iVAR tests ---")
    test_ab_sid_ivar_discrete_forward()
    print()

    # EPIG tests
    print("--- qEPIG tests ---")
    test_qEPIG_forward()
    print()

    # Posterior variance optimization tests
    print("--- Posterior variance optimization tests ---")
    test_fast_pred_var_diagonal_variance_exact()
    test_fast_pred_var_gradient_consistency()
    test_compute_posterior_stats_uses_optimization()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
