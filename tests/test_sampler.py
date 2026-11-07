"""Tests for SMC sampler module."""

import torch

from sidal.sampler import (
    _bisect_next_temperature,
    compute_ess_from_weights,
    systematic_resample,
    sample_smc,
    uniform_reference_point_generator,
    smc_reference_point_generator,
)


# =============================================================================
# Test helper functions
# =============================================================================

def make_gaussian_log_prob(mean, cov):
    """Create log_prob and grad_log_prob for a multivariate Gaussian."""
    mean = torch.as_tensor(mean, dtype=torch.float64)
    cov = torch.as_tensor(cov, dtype=torch.float64)
    prec = torch.linalg.inv(cov)

    def log_prob(x):
        diff = x - mean
        return -0.5 * torch.einsum('...i,...i->...', diff @ prec, diff)

    def grad_log_prob(x):
        diff = x - mean
        return -diff @ prec

    return log_prob, grad_log_prob


def make_gmm_log_prob(means, weights=None):
    """Create log_prob and grad_log_prob for a Gaussian mixture."""
    means = [torch.as_tensor(m, dtype=torch.float64) for m in means]
    n_modes = len(means)
    if weights is None:
        weights = torch.ones(n_modes, dtype=torch.float64) / n_modes
    else:
        weights = torch.as_tensor(weights, dtype=torch.float64)

    def log_prob(x):
        log_ps = []
        for m, w in zip(means, weights):
            log_p = torch.log(w) - 0.5 * ((x - m) ** 2).sum(dim=-1)
            log_ps.append(log_p)
        return torch.logsumexp(torch.stack(log_ps, dim=-1), dim=-1)

    def grad_log_prob(x):
        x_grad = x.clone().requires_grad_(True)
        lp = log_prob(x_grad)
        lp.sum().backward()
        return x_grad.grad

    return log_prob, grad_log_prob


# =============================================================================
# compute_ess_from_weights tests
# =============================================================================

def test_ess_uniform_weights():
    """Test ESS = N for uniform weights."""
    N = 100
    # Uniform log weights (all zeros = equal weights)
    log_weights = torch.zeros(N, dtype=torch.float64)
    ess = compute_ess_from_weights(log_weights)

    assert abs(ess - N) < 1e-10, f"ESS should be {N} for uniform weights, got {ess}"

    print(f"compute_ess_from_weights uniform weights test passed (ESS={ess:.1f})")


def test_ess_degenerate_weights():
    """Test ESS = 1 for degenerate weights (one particle has all weight)."""
    N = 100
    # One very large weight, rest very small
    log_weights = torch.full((N,), -1000.0, dtype=torch.float64)
    log_weights[0] = 0.0  # This particle has essentially all the weight

    ess = compute_ess_from_weights(log_weights)

    assert abs(ess - 1.0) < 1e-6, f"ESS should be 1 for degenerate weights, got {ess}"

    print(f"compute_ess_from_weights degenerate weights test passed (ESS={ess:.6f})")


def test_ess_two_equal_weights():
    """Test ESS = 2 when only two particles have equal weight."""
    N = 100
    log_weights = torch.full((N,), -1000.0, dtype=torch.float64)
    log_weights[0] = 0.0
    log_weights[1] = 0.0  # Two particles share weight equally

    ess = compute_ess_from_weights(log_weights)

    assert abs(ess - 2.0) < 1e-6, f"ESS should be 2 for two equal weights, got {ess}"

    print(f"compute_ess_from_weights two equal weights test passed (ESS={ess:.6f})")


def test_ess_analytical_case():
    """Test ESS against analytical formula for known weight distribution."""
    # Weights proportional to [1, 2, 3, 4] -> normalized [0.1, 0.2, 0.3, 0.4]
    # ESS = 1 / (0.1² + 0.2² + 0.3² + 0.4²) = 1 / 0.30 = 3.333...
    log_weights = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    ess = compute_ess_from_weights(log_weights)

    expected_ess = 1.0 / (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2)
    assert abs(ess - expected_ess) < 1e-10, f"ESS should be {expected_ess:.4f}, got {ess:.4f}"

    print(f"compute_ess_from_weights analytical case test passed (ESS={ess:.4f})")


def test_ess_numerical_stability():
    """Test ESS computation with extreme log weights (numerical stability)."""
    N = 50
    # Very large log weights (would overflow if not handled properly)
    log_weights_large = torch.full((N,), 1000.0, dtype=torch.float64)
    ess_large = compute_ess_from_weights(log_weights_large)

    # Very small log weights
    log_weights_small = torch.full((N,), -1000.0, dtype=torch.float64)
    ess_small = compute_ess_from_weights(log_weights_small)

    # Both should give ESS = N (uniform weights, just shifted)
    assert abs(ess_large - N) < 1e-10, f"ESS should be {N} for shifted uniform, got {ess_large}"
    assert abs(ess_small - N) < 1e-10, f"ESS should be {N} for shifted uniform, got {ess_small}"
    assert not torch.isnan(torch.tensor(ess_large)), "ESS should not be NaN"
    assert not torch.isnan(torch.tensor(ess_small)), "ESS should not be NaN"

    print(f"compute_ess_from_weights numerical stability test passed")


# =============================================================================
# systematic_resample tests
# =============================================================================

def test_resample_output_shape():
    """Test that resampling preserves shape."""
    N, d = 100, 5
    particles = torch.randn(N, d, dtype=torch.float64)
    log_weights = torch.randn(N, dtype=torch.float64)

    resampled = systematic_resample(particles, log_weights)

    assert resampled.shape == (N, d), f"Expected shape ({N}, {d}), got {resampled.shape}"

    print("systematic_resample output shape test passed")


def test_resample_uniform_weights():
    """Test that uniform weights produce each particle exactly once."""
    torch.manual_seed(42)
    N, d = 100, 3
    particles = torch.arange(N, dtype=torch.float64).unsqueeze(1).expand(N, d)
    log_weights = torch.zeros(N, dtype=torch.float64)  # Uniform weights

    resampled = systematic_resample(particles, log_weights)

    # Each particle ID should appear exactly once
    particle_ids = resampled[:, 0].sort()[0]
    expected_ids = torch.arange(N, dtype=torch.float64)

    assert torch.allclose(particle_ids, expected_ids), \
        "With uniform weights, each particle should appear exactly once"

    print("systematic_resample uniform weights test passed")


def test_resample_degenerate_weights():
    """Test that degenerate weights produce all copies of one particle."""
    N, d = 50, 2
    particles = torch.randn(N, d, dtype=torch.float64)
    log_weights = torch.full((N,), -1000.0, dtype=torch.float64)
    log_weights[7] = 0.0  # Particle 7 has all the weight

    resampled = systematic_resample(particles, log_weights)

    # All resampled particles should be particle 7
    expected = particles[7].unsqueeze(0).expand(N, d)
    assert torch.allclose(resampled, expected), \
        "All resampled particles should be the high-weight particle"

    print("systematic_resample degenerate weights test passed")


def test_resample_no_aliasing():
    """Test that modifying resampled particles doesn't affect originals."""
    N, d = 20, 2
    particles = torch.randn(N, d, dtype=torch.float64)
    original_particles = particles.clone()
    log_weights = torch.randn(N, dtype=torch.float64)

    resampled = systematic_resample(particles, log_weights)
    resampled[0, 0] = 999.0  # Modify resampled

    assert torch.allclose(particles, original_particles), \
        "Modifying resampled should not affect original particles"

    print("systematic_resample no aliasing test passed")


def test_resample_statistical_correctness():
    """Test that expected count ≈ N × wᵢ over many resamplings."""
    torch.manual_seed(123)
    N = 100
    d = 1

    # Create particles with known IDs
    particles = torch.arange(N, dtype=torch.float64).unsqueeze(1)

    # Non-uniform weights: particle i has weight proportional to (i+1)
    log_weights = torch.log(torch.arange(1, N + 1, dtype=torch.float64))
    weights_normalized = torch.arange(1, N + 1, dtype=torch.float64)
    weights_normalized = weights_normalized / weights_normalized.sum()

    # Run many resamplings and count
    n_trials = 1000
    counts = torch.zeros(N, dtype=torch.float64)

    for _ in range(n_trials):
        resampled = systematic_resample(particles, log_weights)
        for idx in resampled[:, 0].long():
            counts[idx] += 1

    # Expected counts
    expected_counts = n_trials * N * weights_normalized
    empirical_counts = counts

    # Check correlation (should be very high)
    correlation = torch.corrcoef(torch.stack([expected_counts, empirical_counts]))[0, 1]
    assert correlation > 0.99, f"Correlation should be > 0.99, got {correlation:.4f}"

    # Check relative error for high-weight particles
    high_weight_mask = weights_normalized > 0.01
    relative_error = (empirical_counts[high_weight_mask] - expected_counts[high_weight_mask]).abs() / expected_counts[high_weight_mask]
    max_relative_error = relative_error.max()
    assert max_relative_error < 0.15, f"Max relative error should be < 15%, got {max_relative_error:.1%}"

    print(f"systematic_resample statistical correctness test passed (corr={correlation:.4f})")





# =============================================================================
# _bisect_next_temperature tests
# =============================================================================

def test_bisect_next_temperature_returns_valid_beta():
    """Test that bisection returns β in valid range (β_current, 1]."""
    torch.manual_seed(42)

    # Simple Gaussian target
    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    particles = torch.randn(100, 2, dtype=torch.float64)

    # Test from various starting temperatures
    for beta_current in [0.0, 0.3, 0.5, 0.8]:
        beta_next = _bisect_next_temperature(
            particles, log_prob, beta_current, target_ess_ratio=0.9
        )
        assert beta_next > beta_current, \
            f"β_next ({beta_next}) should be > β_current ({beta_current})"
        assert beta_next <= 1.0, \
            f"β_next ({beta_next}) should be <= 1.0"

    print("_bisect_next_temperature valid range test passed")


def test_bisect_next_temperature_achieves_target_ess():
    """Test that bisection achieves the target ESS ratio."""
    torch.manual_seed(123)

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    N = 200
    particles = torch.randn(N, 3, dtype=torch.float64)
    beta_current = 0.2
    target_ess_ratio = 0.9

    beta_next = _bisect_next_temperature(
        particles, log_prob, beta_current, target_ess_ratio=target_ess_ratio
    )

    # Compute the actual ESS at beta_next
    log_prob_values = log_prob(particles)
    delta_beta = beta_next - beta_current
    log_weights = delta_beta * log_prob_values
    ess = compute_ess_from_weights(log_weights)
    ess_ratio = ess / N

    # ESS ratio should be at or above target (bisection returns conservative β)
    assert ess_ratio >= target_ess_ratio - 0.01, \
        f"ESS ratio {ess_ratio:.3f} should be >= {target_ess_ratio - 0.01}"

    print(f"_bisect_next_temperature target ESS test passed (ESS ratio={ess_ratio:.3f})")


def test_bisect_next_temperature_monotonicity():
    """Test that ESS decreases as Δβ increases (monotonicity property)."""
    torch.manual_seed(456)

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    N = 200
    particles = torch.randn(N, 2, dtype=torch.float64)
    log_prob_values = log_prob(particles)

    # Test ESS at increasing Δβ values
    delta_betas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    ess_values = []

    for delta_beta in delta_betas:
        log_weights = delta_beta * log_prob_values
        ess = compute_ess_from_weights(log_weights)
        ess_values.append(ess)

    # ESS should be monotonically decreasing
    for i in range(len(ess_values) - 1):
        assert ess_values[i] >= ess_values[i + 1], \
            f"ESS should decrease: ESS({delta_betas[i]})={ess_values[i]:.1f} < ESS({delta_betas[i+1]})={ess_values[i+1]:.1f}"

    print(f"_bisect_next_temperature monotonicity test passed")
    print(f"  Δβ: {delta_betas}")
    print(f"  ESS: {[f'{e:.1f}' for e in ess_values]}")


def test_bisect_next_temperature_reaches_one():
    """Test that bisection can return β=1 when ESS stays high."""
    torch.manual_seed(789)

    # Nearly uniform log_prob → weights stay uniform → ESS stays high
    def log_prob(x):
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    particles = torch.randn(100, 2, dtype=torch.float64)

    beta_next = _bisect_next_temperature(
        particles, log_prob, beta_current=0.0, target_ess_ratio=0.9
    )

    assert beta_next == 1.0, \
        f"Should jump directly to β=1 with uniform weights, got {beta_next}"

    print("_bisect_next_temperature reaches β=1 test passed")


def test_bisect_next_temperature_different_targets():
    """Test that lower target ESS gives larger temperature jumps."""
    torch.manual_seed(111)

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    particles = torch.randn(200, 2, dtype=torch.float64)
    beta_current = 0.1

    beta_cautious = _bisect_next_temperature(
        particles, log_prob, beta_current, target_ess_ratio=0.95
    )
    beta_moderate = _bisect_next_temperature(
        particles, log_prob, beta_current, target_ess_ratio=0.8
    )
    beta_aggressive = _bisect_next_temperature(
        particles, log_prob, beta_current, target_ess_ratio=0.5
    )

    # More aggressive (lower ESS target) should give larger jumps
    assert beta_cautious <= beta_moderate <= beta_aggressive, \
        f"Expected β_conservative <= β_moderate <= β_aggressive, " \
        f"got {beta_cautious:.3f}, {beta_moderate:.3f}, {beta_aggressive:.3f}"

    print(f"_bisect_next_temperature different targets test passed")
    print(f"  η*=0.95 → β={beta_cautious:.3f}")
    print(f"  η*=0.80 → β={beta_moderate:.3f}")
    print(f"  η*=0.50 → β={beta_aggressive:.3f}")


def test_bisect_next_temperature_high_dimensional():
    """Test bisection works in higher dimensions (50d)."""
    torch.manual_seed(222)

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    N = 500
    d = 50
    particles = torch.randn(N, d, dtype=torch.float64)

    beta_current = 0.0
    beta_next = _bisect_next_temperature(
        particles, log_prob, beta_current, target_ess_ratio=0.9
    )

    # Verify ESS is achieved
    log_prob_values = log_prob(particles)
    delta_beta = beta_next - beta_current
    log_weights = delta_beta * log_prob_values
    ess = compute_ess_from_weights(log_weights)
    ess_ratio = ess / N

    assert beta_next > 0, f"Should make progress, got β={beta_next}"
    assert ess_ratio >= 0.89, f"ESS ratio {ess_ratio:.3f} should be >= 0.89"

    print(f"_bisect_next_temperature high-d test passed (d={d}, β={beta_next:.3f}, ESS={ess_ratio:.3f})")


# =============================================================================
# sample_smc tests
# =============================================================================

def test_smc_output_shape():
    """Test that SMC returns correct output shapes."""
    torch.manual_seed(42)

    d = 3
    n_particles = 100

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=3, verbose=False
    )

    assert particles.shape == (n_particles, d), \
        f"Expected particles shape ({n_particles}, {d}), got {particles.shape}"
    assert log_weights.shape == (n_particles,), \
        f"Expected log_weights shape ({n_particles},), got {log_weights.shape}"
    assert isinstance(info, dict), "info should be a dict"
    assert "ess_history" in info, "info should contain ess_history"
    assert "accept_history" in info, "info should contain accept_history"

    print("sample_smc output shape test passed")


def test_smc_weights_normalized():
    """Test that final log_weights are normalized (sum to 1 in probability space)."""
    torch.manual_seed(123)

    d = 2
    n_particles = 200

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    _, log_weights, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=3, verbose=False
    )

    # Weights should sum to 1
    weights_sum = torch.exp(log_weights).sum()
    assert torch.allclose(weights_sum, torch.tensor(1.0, dtype=torch.float64), atol=1e-6), \
        f"Weights should sum to 1, got {weights_sum}"

    print("sample_smc weights normalized test passed")


def test_smc_respects_bounds():
    """Test that SMC respects bounds."""
    torch.manual_seed(456)

    d = 3
    n_particles = 200
    lb = torch.zeros(d, dtype=torch.float64)
    ub = torch.ones(d, dtype=torch.float64)
    bounds = (lb, ub)

    # Target centered at (0.5, 0.5, 0.5)
    center = torch.ones(d, dtype=torch.float64) * 0.5

    def log_prob(x):
        return -5.0 * ((x - center) ** 2).sum(dim=-1)

    particles, _, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=bounds, n_mutation_steps=5, verbose=False
    )

    assert (particles >= lb).all(), "Some particles below lower bound"
    assert (particles <= ub).all(), "Some particles above upper bound"

    print("sample_smc respects bounds test passed")


def test_smc_gaussian_2d():
    """Test SMC samples converge to correct Gaussian distribution in 2D."""
    torch.manual_seed(789)

    d = 2
    n_particles = 500
    true_mean = torch.tensor([2.0, -1.0], dtype=torch.float64)
    true_std = torch.tensor([1.0, 0.5], dtype=torch.float64)

    def log_prob(x):
        return -0.5 * (((x - true_mean) / true_std) ** 2).sum(dim=-1)

    # RWM (gradient-free) mixes more slowly than the former MALA kernel; with only
    # 5 mutation steps the particles stay biased toward the N(0, I) initialisation
    # (mean x0 ~ 1.6). 20 RWM steps per temperature recovers the target.
    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=20, verbose=False
    )

    # Compute weighted mean and std
    weights = torch.exp(log_weights)
    weighted_mean = (weights.unsqueeze(-1) * particles).sum(dim=0)
    weighted_var = (weights.unsqueeze(-1) * (particles - weighted_mean) ** 2).sum(dim=0)
    weighted_std = weighted_var.sqrt()

    assert torch.allclose(weighted_mean, true_mean, atol=0.2), \
        f"Weighted mean {weighted_mean.tolist()} not close to true mean {true_mean.tolist()}"
    assert torch.allclose(weighted_std, true_std, rtol=0.3), \
        f"Weighted std {weighted_std.tolist()} not close to true std {true_std.tolist()}"

    print(f"sample_smc Gaussian 2D test passed")
    print(f"  true_mean: {true_mean.tolist()}, weighted_mean: {weighted_mean.tolist()}")
    print(f"  true_std: {true_std.tolist()}, weighted_std: {weighted_std.tolist()}")


def test_smc_gaussian_10d():
    """Test SMC samples converge in 10 dimensions."""
    torch.manual_seed(111)

    d = 10
    n_particles = 1000
    true_mean = torch.arange(d, dtype=torch.float64) * 0.5  # [0, 0.5, 1, 1.5, ...]
    true_std = torch.ones(d, dtype=torch.float64) * 0.8

    def log_prob(x):
        return -0.5 * (((x - true_mean) / true_std) ** 2).sum(dim=-1)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=10, verbose=False
    )

    # Compute weighted mean
    weights = torch.exp(log_weights)
    weighted_mean = (weights.unsqueeze(-1) * particles).sum(dim=0)

    # Check mean is close (allow larger tolerance in higher d)
    mean_error = (weighted_mean - true_mean).abs().max()
    assert mean_error < 0.4, \
        f"Max mean error {mean_error:.3f} too large in 10D"

    print(f"sample_smc Gaussian 10D test passed (max mean error: {mean_error:.3f})")


def test_smc_gaussian_50d():
    """Test SMC samples converge in 50 dimensions."""
    torch.manual_seed(222)

    d = 50
    n_particles = 2000  # Need more particles in high-d
    true_mean = torch.zeros(d, dtype=torch.float64)  # Centered at origin
    true_std = torch.ones(d, dtype=torch.float64)

    def log_prob(x):
        return -0.5 * (((x - true_mean) / true_std) ** 2).sum(dim=-1)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=15,
        verbose=False
    )

    # Compute weighted mean
    weights = torch.exp(log_weights)
    weighted_mean = (weights.unsqueeze(-1) * particles).sum(dim=0)
    weighted_var = (weights.unsqueeze(-1) * (particles - weighted_mean) ** 2).sum(dim=0)
    weighted_std = weighted_var.sqrt()

    # Check statistics
    mean_error = weighted_mean.abs().max()  # Should be near 0
    std_error = (weighted_std - 1.0).abs().max()  # Should be near 1

    assert mean_error < 0.3, f"Max mean error {mean_error:.3f} too large in 50D"
    assert std_error < 0.3, f"Max std error {std_error:.3f} too large in 50D"

    # Check that we actually did multiple temperature steps
    n_temp_steps = info["n_temperature_steps"]
    assert n_temp_steps > 5, f"Expected multiple temperature steps, got {n_temp_steps}"

    print(f"sample_smc Gaussian 50D test passed")
    print(f"  n_temperature_steps: {n_temp_steps}")
    print(f"  n_resamples: {info['n_resamples']}")
    print(f"  max mean error: {mean_error:.3f}")
    print(f"  max std error: {std_error:.3f}")
    print(f"  final ESS ratio: {info['final_ess_ratio']:.2%}")


def test_smc_multimodal():
    """Test SMC can find multiple modes."""
    torch.manual_seed(333)

    d = 2
    n_particles = 500

    # Two modes at (-3, 0) and (3, 0)
    mode1 = torch.tensor([-3.0, 0.0], dtype=torch.float64)
    mode2 = torch.tensor([3.0, 0.0], dtype=torch.float64)

    def log_prob(x):
        log_p1 = -0.5 * ((x - mode1) ** 2).sum(dim=-1)
        log_p2 = -0.5 * ((x - mode2) ** 2).sum(dim=-1)
        return torch.logsumexp(torch.stack([log_p1, log_p2], dim=-1), dim=-1)

    particles, log_weights, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=5, verbose=False
    )

    # Count particles in each mode
    near_mode1 = (particles[:, 0] < 0).sum().item()
    near_mode2 = (particles[:, 0] > 0).sum().item()
    pct1 = near_mode1 / n_particles
    pct2 = near_mode2 / n_particles

    # Both modes should have reasonable representation
    assert pct1 > 0.2, f"Mode 1 has only {pct1:.1%} of particles"
    assert pct2 > 0.2, f"Mode 2 has only {pct2:.1%} of particles"

    print(f"sample_smc multimodal test passed (mode1: {pct1:.1%}, mode2: {pct2:.1%})")


# -----------------------------------------------------------------------------
# GMM (5-component) tests
# -----------------------------------------------------------------------------

def _make_gmm_target(means, std=0.5, weights=None):
    """
    Create a K-component GMM target.

    Args:
        means: List of K mean vectors, each shape (d,)
        std: Shared standard deviation for all components
        weights: Optional mixture weights (must sum to 1). If None, uniform weights.

    Returns:
        log_prob function (supports automatic differentiation)
    """
    means = [torch.as_tensor(m, dtype=torch.float64) for m in means]
    K = len(means)
    var = std ** 2

    if weights is None:
        weights = torch.ones(K, dtype=torch.float64) / K
    else:
        weights = torch.as_tensor(weights, dtype=torch.float64)
        weights = weights / weights.sum()  # Normalize

    log_weights = torch.log(weights)

    def log_prob(x):
        # x: (N, d), compute log sum_k w_k * N(x; mu_k, std^2 I)
        log_components = []
        for mu, log_w in zip(means, log_weights):
            log_p = log_w - 0.5 * ((x - mu) ** 2).sum(dim=-1) / var
            log_components.append(log_p)
        return torch.logsumexp(torch.stack(log_components, dim=-1), dim=-1)

    return log_prob


def _count_modes(particles, means, threshold=2.0):
    """Count particles near each mode."""
    means = [torch.as_tensor(m, dtype=torch.float64) for m in means]
    counts = []
    for mu in means:
        dist = ((particles - mu) ** 2).sum(dim=-1).sqrt()
        count = (dist < threshold).sum().item()
        counts.append(count)
    return counts


def test_smc_gmm_2d():
    """Test SMC finds all 5 modes in 2D GMM."""
    torch.manual_seed(555)

    d = 2
    n_particles = 1000

    # 5 modes arranged in a cross pattern
    means = [
        [0.0, 0.0],
        [4.0, 0.0],
        [-4.0, 0.0],
        [0.0, 4.0],
        [0.0, -4.0],
    ]
    std = 0.8

    # Bounds that cover all modes uniformly
    lb = torch.tensor([-6.0, -6.0], dtype=torch.float64)
    ub = torch.tensor([6.0, 6.0], dtype=torch.float64)
    bounds = (lb, ub)

    log_prob = _make_gmm_target(means, std=std)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=bounds, n_mutation_steps=5, verbose=False
    )

    # Count particles near each mode
    counts = _count_modes(particles, means, threshold=2.0)
    total_near_modes = sum(counts)
    pcts = [c / n_particles for c in counts]

    # All modes should have at least 5% of particles
    min_pct = min(pcts)
    assert min_pct > 0.05, f"Some mode has only {min_pct:.1%} particles: {pcts}"

    # Most particles should be near some mode
    assert total_near_modes / n_particles > 0.8, \
        f"Only {total_near_modes / n_particles:.1%} particles near modes"

    print(f"sample_smc GMM 2D test passed")
    print(f"  mode proportions: {[f'{p:.1%}' for p in pcts]}")
    print(f"  particles near modes: {total_near_modes / n_particles:.1%}")


def test_smc_gmm_10d():
    """Test SMC finds all 5 modes in 10D GMM."""
    torch.manual_seed(666)

    d = 10
    n_particles = 2000

    # 5 modes along different axes
    means = []
    for i in range(5):
        mu = torch.zeros(d, dtype=torch.float64)
        mu[i % d] = 4.0 * (1 if i % 2 == 0 else -1)
        means.append(mu.tolist())

    std = 1.0

    # Bounds that cover all modes
    lb = torch.full((d,), -6.0, dtype=torch.float64)
    ub = torch.full((d,), 6.0, dtype=torch.float64)
    bounds = (lb, ub)

    log_prob = _make_gmm_target(means, std=std)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=bounds, n_mutation_steps=10, verbose=False
    )

    # Count particles near each mode
    counts = _count_modes(particles, means, threshold=3.0)
    total_near_modes = sum(counts)
    pcts = [c / n_particles for c in counts]

    # Check mode coverage
    modes_found = sum(1 for p in pcts if p > 0.03)

    assert modes_found >= 4, f"Only found {modes_found}/5 modes: {pcts}"

    print(f"sample_smc GMM 10D test passed")
    print(f"  modes found: {modes_found}/5")
    print(f"  mode proportions: {[f'{p:.1%}' for p in pcts]}")
    print(f"  n_temperature_steps: {info['n_temperature_steps']}")


def test_smc_gmm_50d():
    """Test SMC finds modes in 50D GMM."""
    torch.manual_seed(777)

    d = 50
    n_particles = 3000

    # 5 modes spread in different subspaces
    means = []
    for i in range(5):
        mu = torch.zeros(d, dtype=torch.float64)
        start_idx = i * 10
        for j in range(10):
            mu[(start_idx + j) % d] = 3.0
        means.append(mu.tolist())

    std = 1.5

    # Bounds that cover all modes
    lb = torch.full((d,), -5.0, dtype=torch.float64)
    ub = torch.full((d,), 8.0, dtype=torch.float64)
    bounds = (lb, ub)

    log_prob = _make_gmm_target(means, std=std)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        # RWM (gradient-free) needs more mutation steps than the former MALA
        # kernel to rebalance mass across modes in 50D (20 steps: max mean error ~1.8).
        bounds=bounds, n_mutation_steps=40,
        verbose=False
    )

    # In 50D, check weighted mean and particle spread
    weights = torch.exp(log_weights)
    weighted_mean = (weights.unsqueeze(-1) * particles).sum(dim=0)

    # True mean of GMM
    true_mean = torch.stack([torch.tensor(m, dtype=torch.float64) for m in means]).mean(dim=0)
    mean_error = (weighted_mean - true_mean).abs().max()

    # Check particles are spread (not collapsed)
    particle_std = particles.std(dim=0).mean()

    assert mean_error < 1.0, f"Max mean error {mean_error:.3f} too large in 50D GMM"
    assert particle_std > 0.5, f"Particles too collapsed: std={particle_std:.3f}"

    print(f"sample_smc GMM 50D test passed")
    print(f"  n_temperature_steps: {info['n_temperature_steps']}")
    print(f"  n_resamples: {info['n_resamples']}")
    print(f"  max mean error: {mean_error:.3f}")
    print(f"  particle spread (avg std): {particle_std:.3f}")
    print(f"  final ESS ratio: {info['final_ess_ratio']:.2%}")


def test_smc_gmm_imbalanced_weights():
    """Test SMC samples proportionally to imbalanced GMM weights."""
    torch.manual_seed(888)

    d = 2
    n_particles = 2000

    # 5 modes with imbalanced weights
    means = [
        [0.0, 0.0],    # weight 0.4
        [5.0, 0.0],    # weight 0.3
        [-5.0, 0.0],   # weight 0.15
        [0.0, 5.0],    # weight 0.1
        [0.0, -5.0],   # weight 0.05
    ]
    true_weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    std = 0.8

    # Bounds covering all modes
    lb = torch.tensor([-7.0, -7.0], dtype=torch.float64)
    ub = torch.tensor([7.0, 7.0], dtype=torch.float64)
    bounds = (lb, ub)

    log_prob = _make_gmm_target(means, std=std, weights=true_weights)

    particles, log_weights, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=bounds, n_mutation_steps=10, verbose=False
    )

    # Count particles near each mode
    counts = _count_modes(particles, means, threshold=2.0)
    total_near_modes = sum(counts)
    empirical_pcts = [c / n_particles for c in counts]

    # Check correlation between true weights and empirical proportions
    true_w = torch.tensor(true_weights, dtype=torch.float64)
    emp_w = torch.tensor(empirical_pcts, dtype=torch.float64)
    correlation = torch.corrcoef(torch.stack([true_w, emp_w]))[0, 1]

    # Correlation should be high (sampling proportional to weights)
    assert correlation > 0.9, \
        f"Correlation between true weights and empirical should be > 0.9, got {correlation:.3f}"

    # Dominant mode (40%) should have most particles
    assert empirical_pcts[0] > 0.25, \
        f"Dominant mode should have >25% particles, got {empirical_pcts[0]:.1%}"

    # Smallest mode (5%) should have fewest
    assert empirical_pcts[4] < empirical_pcts[0], \
        f"Smallest mode should have fewer particles than dominant"

    print(f"sample_smc GMM imbalanced weights test passed")
    print(f"  true weights:     {[f'{w:.0%}' for w in true_weights]}")
    print(f"  empirical pcts:   {[f'{p:.1%}' for p in empirical_pcts]}")
    print(f"  correlation: {correlation:.3f}")


def test_smc_info_diagnostics():
    """Test that SMC returns useful diagnostic information."""
    torch.manual_seed(444)

    d = 3
    n_particles = 200

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    _, _, info = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        n_mutation_steps=3, verbose=False
    )

    # Check all expected keys are present
    expected_keys = [
        "n_particles", "n_temperature_steps", "n_resamples",
        "final_ess", "final_ess_ratio", "final_step_size",
        "ess_history", "accept_history", "temperature_history"
    ]
    for key in expected_keys:
        assert key in info, f"Missing key '{key}' in info dict"

    # Check values are reasonable
    assert info["n_particles"] == n_particles
    assert info["n_temperature_steps"] > 0
    assert 0 <= info["final_ess_ratio"] <= 1
    assert len(info["ess_history"]) == info["n_temperature_steps"]
    assert len(info["temperature_history"]) == info["n_temperature_steps"] + 1
    assert info["temperature_history"][-1] == 1.0  # Should reach β=1

    print(f"sample_smc info diagnostics test passed")
    print(f"  n_temperature_steps: {info['n_temperature_steps']}")
    print(f"  n_resamples: {info['n_resamples']}")
    print(f"  final_ess_ratio: {info['final_ess_ratio']:.2%}")


def test_smc_reproducibility_with_seed():
    """Test that SMC with same seed gives identical results."""
    d = 2
    n_particles = 100

    def log_prob(x):
        return -0.5 * (x ** 2).sum(dim=-1)

    lb = torch.tensor([-3.0, -3.0], dtype=torch.float64)
    ub = torch.tensor([3.0, 3.0], dtype=torch.float64)

    # Run twice with same seed
    particles1, log_w1, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=(lb, ub), n_mutation_steps=3, seed=12345, verbose=False
    )
    particles2, log_w2, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=(lb, ub), n_mutation_steps=3, seed=12345, verbose=False
    )

    # Should be identical
    assert torch.allclose(particles1, particles2), "Same seed should give identical particles"
    assert torch.allclose(log_w1, log_w2), "Same seed should give identical weights"

    # Run with different seed
    particles3, _, _ = sample_smc(
        log_prob, d=d, n_particles=n_particles,
        bounds=(lb, ub), n_mutation_steps=3, seed=99999, verbose=False
    )

    # Should be different
    assert not torch.allclose(particles1, particles3), "Different seed should give different particles"

    print("sample_smc reproducibility with seed test passed")


# =============================================================================
# uniform_reference_point_generator tests
# =============================================================================

def test_uniform_generator_output_shape():
    """Test output shape is (n_samples, d)."""
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    pts = uniform_reference_point_generator(bounds, n_samples=100, seed=0)
    assert pts.shape == (100, 3), f"Expected (100, 3), got {pts.shape}"
    print("uniform_reference_point_generator output shape test passed")


def test_uniform_generator_within_bounds():
    """Test all points are within bounds."""
    lb = torch.tensor([-2.0, 0.0], dtype=torch.float64)
    ub = torch.tensor([3.0, 5.0], dtype=torch.float64)
    bounds = torch.stack([lb, ub])
    pts = uniform_reference_point_generator(bounds, n_samples=1000, seed=42)

    assert (pts[:, 0] >= -2.0).all(), "Points below lower bound in dim 0"
    assert (pts[:, 0] <= 3.0).all(), "Points above upper bound in dim 0"
    assert (pts[:, 1] >= 0.0).all(), "Points below lower bound in dim 1"
    assert (pts[:, 1] <= 5.0).all(), "Points above upper bound in dim 1"
    print("uniform_reference_point_generator bounds test passed")


def test_uniform_generator_reproducibility():
    """Test same seed gives same points."""
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    pts1 = uniform_reference_point_generator(bounds, n_samples=50, seed=123)
    pts2 = uniform_reference_point_generator(bounds, n_samples=50, seed=123)

    assert torch.allclose(pts1, pts2), "Same seed should give identical points"
    print("uniform_reference_point_generator reproducibility test passed")


def test_uniform_generator_different_seeds():
    """Test different seeds give different points."""
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

    pts1 = uniform_reference_point_generator(bounds, n_samples=50, seed=0)
    pts2 = uniform_reference_point_generator(bounds, n_samples=50, seed=999)

    assert not torch.allclose(pts1, pts2), "Different seeds should give different points"
    print("uniform_reference_point_generator different seeds test passed")


def test_uniform_generator_coverage():
    """Test points cover the space reasonably uniformly."""
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    pts = uniform_reference_point_generator(bounds, n_samples=1000, seed=42)

    # Check mean is close to 0.5 (center of [0, 1])
    mean = pts.mean().item()
    assert abs(mean - 0.5) < 0.05, f"Mean {mean} too far from expected 0.5"

    # Check std is close to 1/sqrt(12) ≈ 0.289 for uniform [0, 1]
    std = pts.std().item()
    expected_std = 1.0 / (12 ** 0.5)
    assert abs(std - expected_std) < 0.03, f"Std {std} too far from expected {expected_std}"

    print(f"uniform_reference_point_generator coverage test passed (mean={mean:.3f}, std={std:.3f})")


# =============================================================================
# smc_reference_point_generator tests
# =============================================================================

class DummyEnergyDistribution:
    """Dummy energy distribution for testing smc_reference_point_generator.

    Mimics the interface of BoltzmannEnergyDistribution by providing
    compute_log_unnorm_density(x, stabilize=...) -> log probability.
    """
    def __init__(self, mean, std=1.0):
        """Create a Gaussian energy distribution centered at mean."""
        self.mean = torch.as_tensor(mean, dtype=torch.float64)
        self.std = std
        self.var = std ** 2

    def compute_log_unnorm_density(self, x, stabilize=False):
        """Log probability of Gaussian: -0.5 * ||x - mean||^2 / var."""
        diff = x - self.mean
        return -0.5 * (diff ** 2).sum(dim=-1) / self.var


class DummyGMMEnergyDistribution:
    """Dummy GMM energy distribution for testing multimodal sampling."""
    def __init__(self, means, std=0.5, weights=None):
        self.means = [torch.as_tensor(m, dtype=torch.float64) for m in means]
        self.std = std
        self.var = std ** 2
        K = len(means)
        if weights is None:
            self.weights = torch.ones(K, dtype=torch.float64) / K
        else:
            self.weights = torch.as_tensor(weights, dtype=torch.float64)
            self.weights = self.weights / self.weights.sum()
        self.log_weights = torch.log(self.weights)

    def compute_log_unnorm_density(self, x, stabilize=False):
        log_components = []
        for mu, log_w in zip(self.means, self.log_weights):
            log_p = log_w - 0.5 * ((x - mu) ** 2).sum(dim=-1) / self.var
            log_components.append(log_p)
        return torch.logsumexp(torch.stack(log_components, dim=-1), dim=-1)


def test_smc_generator_output_shape():
    """Test output shape is (n_samples, d)."""
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    energy_dist = DummyEnergyDistribution(mean=[0.5, 0.5])

    pts = smc_reference_point_generator(
        energy_dist, bounds, n_samples=100, n_mutation_steps=3, seed=0, verbose=False
    )

    assert pts.shape == (100, 2), f"Expected (100, 2), got {pts.shape}"
    print("smc_reference_point_generator output shape test passed")


def test_smc_generator_within_bounds():
    """Test all points are within bounds."""
    bounds = torch.tensor([[-2.0, -1.0], [2.0, 3.0]], dtype=torch.float64)
    energy_dist = DummyEnergyDistribution(mean=[0.0, 1.0])

    pts = smc_reference_point_generator(
        energy_dist, bounds, n_samples=500, n_mutation_steps=5, seed=42, verbose=False
    )

    assert (pts[:, 0] >= -2.0).all(), "Points below lower bound in dim 0"
    assert (pts[:, 0] <= 2.0).all(), "Points above upper bound in dim 0"
    assert (pts[:, 1] >= -1.0).all(), "Points below lower bound in dim 1"
    assert (pts[:, 1] <= 3.0).all(), "Points above upper bound in dim 1"
    print("smc_reference_point_generator bounds test passed")


def test_smc_generator_samples_near_mode():
    """Test samples concentrate near the target mode."""
    mean = [2.0, -1.0]
    bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.float64)
    energy_dist = DummyEnergyDistribution(mean=mean, std=0.5)

    pts = smc_reference_point_generator(
        energy_dist, bounds, n_samples=500, n_mutation_steps=10, seed=123, verbose=False
    )

    # Check sample mean is close to target
    sample_mean = pts.mean(dim=0)
    mean_error = (sample_mean - torch.tensor(mean, dtype=torch.float64)).abs().max().item()
    assert mean_error < 0.3, f"Sample mean {sample_mean.tolist()} too far from {mean}"

    # Check sample std is close to target
    sample_std = pts.std(dim=0)
    std_error = (sample_std - 0.5).abs().max().item()
    assert std_error < 0.15, f"Sample std {sample_std.tolist()} too far from 0.5"

    print(f"smc_reference_point_generator samples near mode test passed")
    print(f"  target mean: {mean}, sample mean: {sample_mean.tolist()}")
    print(f"  target std: 0.5, sample std: {sample_std.tolist()}")


def test_smc_generator_multimodal():
    """Test SMC generator finds multiple modes."""
    means = [[-3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]
    bounds = torch.tensor([[-6.0, -3.0], [6.0, 6.0]], dtype=torch.float64)
    energy_dist = DummyGMMEnergyDistribution(means=means, std=0.5)

    pts = smc_reference_point_generator(
        energy_dist, bounds, n_samples=600, n_mutation_steps=10, seed=456, verbose=False
    )

    # Count particles near each mode
    threshold = 1.5
    counts = []
    for m in means:
        m_tensor = torch.tensor(m, dtype=torch.float64)
        dist = ((pts - m_tensor) ** 2).sum(dim=-1).sqrt()
        count = (dist < threshold).sum().item()
        counts.append(count)

    # All modes should have some particles
    for i, count in enumerate(counts):
        assert count > 50, f"Mode {i} has only {count} particles (expected > 50)"

    total_near_modes = sum(counts)
    coverage = total_near_modes / 600

    print(f"smc_reference_point_generator multimodal test passed")
    print(f"  mode counts: {counts}")
    print(f"  coverage: {coverage:.1%}")


def test_smc_generator_reproducibility():
    """Test same seed gives same points."""
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    energy_dist = DummyEnergyDistribution(mean=[0.5, 0.5])

    pts1 = smc_reference_point_generator(
        energy_dist, bounds, n_samples=100, n_mutation_steps=3, seed=999, verbose=False
    )
    pts2 = smc_reference_point_generator(
        energy_dist, bounds, n_samples=100, n_mutation_steps=3, seed=999, verbose=False
    )
    pts3 = smc_reference_point_generator(
        energy_dist, bounds, n_samples=100, n_mutation_steps=3, seed=123, verbose=False
    )

    assert torch.allclose(pts1, pts2), "Same seed should give identical points"
    assert not torch.allclose(pts1, pts3), "Different seed should give different points"

    print("smc_reference_point_generator reproducibility test passed")


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all sampler tests."""
    print("=" * 60)
    print("Testing sampler module")
    print("=" * 60)
    print()

    print("--- compute_ess_from_weights tests ---")
    test_ess_uniform_weights()
    test_ess_degenerate_weights()
    test_ess_two_equal_weights()
    test_ess_analytical_case()
    test_ess_numerical_stability()
    print()

    print("--- systematic_resample tests ---")
    test_resample_output_shape()
    test_resample_uniform_weights()
    test_resample_degenerate_weights()
    test_resample_no_aliasing()
    test_resample_statistical_correctness()
    print()

    print()

    print("--- _bisect_next_temperature tests ---")
    test_bisect_next_temperature_returns_valid_beta()
    test_bisect_next_temperature_achieves_target_ess()
    test_bisect_next_temperature_monotonicity()
    test_bisect_next_temperature_reaches_one()
    test_bisect_next_temperature_different_targets()
    test_bisect_next_temperature_high_dimensional()
    print()

    print("--- sample_smc tests ---")
    test_smc_output_shape()
    test_smc_weights_normalized()
    test_smc_respects_bounds()
    test_smc_gaussian_2d()
    test_smc_gaussian_10d()
    test_smc_gaussian_50d()
    test_smc_multimodal()
    test_smc_gmm_2d()
    test_smc_gmm_10d()
    test_smc_gmm_50d()
    test_smc_gmm_imbalanced_weights()
    test_smc_info_diagnostics()
    test_smc_reproducibility_with_seed()
    print()

    print("--- uniform_reference_point_generator tests ---")
    test_uniform_generator_output_shape()
    test_uniform_generator_within_bounds()
    test_uniform_generator_reproducibility()
    test_uniform_generator_different_seeds()
    test_uniform_generator_coverage()
    print()

    print("--- smc_reference_point_generator tests ---")
    test_smc_generator_output_shape()
    test_smc_generator_within_bounds()
    test_smc_generator_samples_near_mode()
    test_smc_generator_multimodal()
    test_smc_generator_reproducibility()
    print()

    print("=" * 60)
    print("All sampler tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
