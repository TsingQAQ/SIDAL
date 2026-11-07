"""
SMC sampler with Random Walk Metropolis (RWM) mutations.

Sequential Monte Carlo sampler using tempering + RWM rejuvenation steps.
"""

import time
import torch
from typing import Callable, Optional, Tuple, Dict, List, Union


def compute_ess_from_weights(log_weights: torch.Tensor) -> float:
    """
    Compute effective sample size from log weights.
    
    Args:
        log_weights: (N,) unnormalized log weights
    
    Returns:
        ESS as a float
    """
    # Normalize in log space for numerical stability
    log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights_normalized)
    ess = 1.0 / (weights ** 2).sum()
    return ess.item()


def systematic_resample(
    particles: torch.Tensor,
    log_weights: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Systematic resampling with lower variance than multinomial resampling.

    Resampling addresses weight degeneracy in SMC by duplicating high-weight
    particles and discarding low-weight ones, producing N equally-weighted
    particles that approximate the same distribution.

    Algorithm:
        1. Normalize weights: wᵢ = exp(log_wᵢ) / Σ exp(log_wⱼ)
        2. Compute CDF: F(i) = Σⱼ≤ᵢ wⱼ
        3. Generate evenly-spaced positions with random offset:
           u₀ ~ Uniform(0, 1/N), then uₖ = u₀ + k/N for k = 0, ..., N-1
        4. Select particle i if uₖ ∈ [F(i-1), F(i))

    Why systematic over multinomial:
        - Multinomial: Each position sampled independently → high variance
        - Systematic: Stratified positions → lower variance, better coverage
        - If wᵢ = 1/N (uniform), systematic gives exactly one copy of each
          particle, while multinomial would have random duplicates/gaps

    Note:
        After resampling, all particles have equal weight (1/N). The returned
        particles are cloned to avoid aliasing issues.

    Args:
        particles: Current particle positions, shape (N, d).
        log_weights: Unnormalized log importance weights, shape (N,).
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Resampled particles with shape (N, d), equally weighted.
    """
    N = particles.shape[0]
    device = particles.device
    dtype = particles.dtype

    # Normalize weights
    log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights_normalized)

    # Cumulative sum
    cumsum = torch.cumsum(weights, dim=0)

    # Systematic resampling positions
    u0 = torch.rand(1, device=device, dtype=dtype, generator=generator) / N
    u = u0 + torch.arange(N, device=device, dtype=dtype) / N

    # Find indices
    indices = torch.searchsorted(cumsum, u)
    indices = indices.clamp(max=N - 1)

    return particles[indices].clone()


def rwm_mutation_step(
    particles: torch.Tensor,
    log_prob_tempered: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    precond: torch.Tensor,
    bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Single Random Walk Metropolis (RWM) step for all particles.

    RWM proposes moves from a symmetric Gaussian proposal, without using
    gradient information.

    Proposal distribution:
        x' = x + σ × P × z,  where z ~ N(0, I)

    where σ is step_size and P is the diagonal preconditioning matrix.

    Acceptance probability (Metropolis-Hastings):
        α = min(1, π(x') / π(x))

    Note:
        - No gradients required (faster GP evaluation)
        - Proposals outside bounds are automatically rejected
        - All N particles are updated in parallel (vectorized)

    Args:
        particles: Current particle positions, shape (N, d).
        log_prob_tempered: Function computing log π_β(x). No gradients needed.
        step_size: RWM step size σ.
        precond: Diagonal preconditioning, shape (d,).
        bounds: Optional (lower_bound, upper_bound) tensors, each shape (d,).
        generator: Optional torch.Generator for reproducibility.

    Returns:
        new_particles: Updated particle positions, shape (N, d).
        accept_rate: Fraction of proposals accepted, in [0, 1].
    """
    device = particles.device
    dtype = particles.dtype
    N = particles.shape[0]

    # Current log_prob
    log_prob_current = log_prob_tempered(particles)

    # Propose (symmetric Gaussian)
    noise = torch.randn(particles.shape, device=device, dtype=dtype, generator=generator)
    proposed = particles + step_size * precond * noise

    # Handle bounds
    if bounds is not None:
        lb, ub = bounds
        lb = lb.to(device=device, dtype=dtype)
        ub = ub.to(device=device, dtype=dtype)
        outside_mask = torch.any((proposed < lb) | (proposed > ub), dim=-1)
    else:
        outside_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Proposed log_prob
    log_prob_proposed = log_prob_tempered(proposed)

    # MH ratio (symmetric proposal, so just likelihood ratio)
    log_alpha = log_prob_proposed - log_prob_current
    log_alpha = torch.clamp(log_alpha, max=0.0)

    # Accept/reject
    log_u = torch.log(torch.rand(N, device=device, dtype=dtype, generator=generator))
    accept = (log_u < log_alpha) & (~outside_mask)

    new_particles = torch.where(accept.unsqueeze(-1), proposed, particles)
    accept_rate = accept.float().mean().item()

    return new_particles, accept_rate


def sample_smc(
    log_unnorm_prob: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    n_particles: int = 1000,
    n_temperatures: int = 20,
    n_mutation_steps: int = 10,
    mutation_step_size: float = 0.1,
    ess_threshold: float = 0.5,
    target_ess_ratio: float = 0.9,
    bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    temperature_schedule: Optional[List[float]] = None,
    adaptive_temperatures: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    verbose: bool = False,
    mutation_kernel: str = "rwm",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Annealed SMC sampler with RWM mutations for sampling from exp(log_unnorm_prob(x)).

    This is an SMC sampler (Sequential Monte Carlo) that uses temperature annealing
    to sample from static distributions. The algorithm constructs a sequence of
    tempered distributions: π_β(x) ∝ exp(β * log_unnorm_prob(x)), with β: 0 → 1,
    transitioning from uniform (β=0) to the target (β=1).

    Algorithm:
        1. Initialize particles uniformly (or from prior)
        2. For each temperature step:
           a. Reweight particles for temperature change
           b. Resample if ESS drops below threshold (systematic resampling)
           c. Apply RWM mutations to rejuvenate particles

    Temperature schedule options:
        - Adaptive (default): Uses binary search to find the largest Δβ such that
          ESS remains above `target_ess_ratio * n_particles`. This is based on the
          fact that ESS is monotonically decreasing in Δβ: larger temperature jumps
          cause more weight variance, thus lower ESS. Self-tuning, no schedule needed.
        - Fixed: Provide explicit schedule via `temperature_schedule` parameter
          (e.g., linear, geometric, cosine).

    The normalization constant of the target is not required since SMC uses
    importance weight ratios where the constant cancels out.

    Args:
        log_unnorm_prob: Unnormalized log density of target, (N, d) -> (N,).
            Only needs to be correct up to an additive constant.
        d: Dimension of the sampling space.
        n_particles: Number of particles to use.
        n_temperatures: Number of temperature steps (if not adaptive).
        n_mutation_steps: RWM mutation steps per temperature level.
        mutation_step_size: Initial step size (adapted during sampling).
        ess_threshold: Resample when ESS < threshold * n_particles.
        target_ess_ratio: Target ESS ratio for adaptive temperature selection.
            At each step, find the largest Δβ such that ESS ≥ target_ess_ratio * N.
            Higher values (e.g., 0.9) give smaller, safer steps; lower values
            (e.g., 0.5) give larger, more aggressive steps. Default: 0.9.
        bounds: Optional (lb, ub) tuple, each of shape (d,).
        temperature_schedule: Optional explicit [β₀, β₁, ..., βₖ] schedule.
        adaptive_temperatures: If True, adaptively choose temperatures to
            maintain target ESS ratio (default: True).
        device: Torch device.
        dtype: Torch dtype.
        seed: Optional random seed for full reproducibility. If None, uses
            global torch random state.
        verbose: Print progress information.
        mutation_kernel: Mutation kernel; only "rwm" (Random Walk Metropolis) is supported.

    Returns:
        particles: (n_particles, d) final particle positions.
        log_weights: (n_particles,) final normalized log weights.
        info: Dict with diagnostics (ESS history, acceptance rates, etc.).
    """
    if device is None:
        device = torch.device("cpu")

    # Create generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Initialize particles from prior (uniform if bounded, standard normal otherwise)
    if bounds is not None:
        lb, ub = bounds
        lb = lb.to(device=device, dtype=dtype)
        ub = ub.to(device=device, dtype=dtype)
        particles = torch.rand(n_particles, d, device=device, dtype=dtype, generator=generator) * (ub - lb) + lb
    else:
        particles = torch.randn(n_particles, d, device=device, dtype=dtype, generator=generator)

    # Initialize uniform weights (in log space)
    log_weights = torch.zeros(n_particles, device=device, dtype=dtype)
    
    # Preconditioning (will be adapted)
    precond = torch.ones(d, device=device, dtype=dtype)
    step_size = mutation_step_size
    
    # Temperature schedule
    if temperature_schedule is not None:
        betas = temperature_schedule
    elif adaptive_temperatures:
        betas = [0.0]  # Will be built adaptively
    else:
        betas = torch.linspace(0, 1, n_temperatures + 1).tolist()
    
    # Tracking
    ess_history = []
    accept_history = []
    temperature_history = [0.0]
    n_resamples = 0
    
    beta_current = 0.0
    step_idx = 0
    
    if verbose:
        print(f"SMC Sampler: d={d}, n_particles={n_particles}")
        print("Starting annealing from beta=0 to beta=1...")

    # Start timing
    smc_start_time = time.time()

    while beta_current < 1.0:
        # Determine next temperature
        if adaptive_temperatures and temperature_schedule is None:
            # Binary search for next β that gives target ESS
            beta_next = _bisect_next_temperature(
                particles, log_unnorm_prob, beta_current,
                target_ess_ratio=target_ess_ratio,
            )
        else:
            if step_idx + 1 < len(betas):
                beta_next = betas[step_idx + 1]
            else:
                beta_next = 1.0
        
        beta_next = min(beta_next, 1.0)
        delta_beta = beta_next - beta_current
        
        # (a) Reweight for temperature change
        with torch.no_grad():
            log_prob_values = log_unnorm_prob(particles)
        log_weights = log_weights + delta_beta * log_prob_values
        
        # Compute ESS
        ess = compute_ess_from_weights(log_weights)
        ess_ratio = ess / n_particles
        ess_history.append(ess_ratio)
        
        if verbose:
            print(f"  Step {step_idx + 1}: beta={beta_next:.4f}, ESS={ess_ratio:.2%}")
        
        # (b) Resample if ESS too low
        if ess_ratio < ess_threshold:
            particles = systematic_resample(particles, log_weights, generator=generator)
            log_weights = torch.zeros(n_particles, device=device, dtype=dtype)
            n_resamples += 1

            if verbose:
                print(f"    Resampled (ESS was {ess_ratio:.2%})")

        # (c) RWM mutation at current temperature
        step_accepts = []

        if mutation_kernel == "rwm":
            # RWM: No gradients needed
            def log_prob_tempered(x):
                with torch.no_grad():
                    return beta_next * log_unnorm_prob(x)

            for _ in range(n_mutation_steps):
                particles, accept_rate = rwm_mutation_step(
                    particles, log_prob_tempered,
                    step_size, precond, bounds, generator=generator
                )
                step_accepts.append(accept_rate)

            # Adapt step size (RWM targets ~23-44% acceptance)
            avg_accept = sum(step_accepts) / len(step_accepts)
            if avg_accept < 0.15:
                step_size *= 0.8
            elif avg_accept > 0.5:
                step_size *= 1.2
        else:
            raise ValueError(f"Unknown mutation_kernel: {mutation_kernel}. Only 'rwm' is supported.")

        accept_history.append(avg_accept)
        step_size = max(1e-6, min(step_size, 10.0))
        
        # Update preconditioner periodically
        if (step_idx + 1) % 5 == 0:
            precond = particles.std(dim=0).clamp(min=1e-6)
        
        beta_current = beta_next
        temperature_history.append(beta_current)
        step_idx += 1
        
        # Safety check
        if step_idx > 1000:
            if verbose:
                print("Warning: exceeded 1000 temperature steps")
            break
    
    # End timing
    smc_total_time = time.time() - smc_start_time

    # Final normalization of weights
    log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)

    # Final ESS
    final_ess = compute_ess_from_weights(log_weights)

    # Always print SMC timing summary
    print(f"SMC done: {smc_total_time:.2f}s, {step_idx} steps, ESS={final_ess / n_particles:.0%}, kernel={mutation_kernel}")

    if verbose:
        print(f"\nSMC complete:")
        print(f"  Temperature steps: {step_idx}")
        print(f"  Resampling events: {n_resamples}")
        print(f"  Final ESS: {final_ess / n_particles:.2%}")
        print(f"  Final step size: {step_size:.6f}")
        print(f"  Total time: {smc_total_time:.2f}s")

    info = {
        "n_particles": n_particles,
        "n_temperature_steps": step_idx,
        "n_resamples": n_resamples,
        "final_ess": final_ess,
        "final_ess_ratio": final_ess / n_particles,
        "final_step_size": step_size,
        "ess_history": ess_history,
        "accept_history": accept_history,
        "temperature_history": temperature_history,
        "total_time": smc_total_time,
        "mutation_kernel": mutation_kernel,
    }
    
    return particles, log_weights_normalized, info


def print_smc_diagnostics(info: Dict) -> None:
    """Pretty print SMC diagnostics.

    Args:
        info: Diagnostics dict returned by sample_smc.
    """
    print("\n" + "=" * 60)
    print("SMC DIAGNOSTICS")
    print("=" * 60)

    print(f"\nSampling Summary:")
    print(f"  Particles:          {info['n_particles']}")
    print(f"  Temperature steps:  {info['n_temperature_steps']}")
    print(f"  Resampling events:  {info['n_resamples']}")
    if "mutation_kernel" in info:
        print(f"  Mutation kernel:    {info['mutation_kernel'].upper()}")

    print(f"\nFinal State:")
    print(f"  ESS:                {info['final_ess']:.1f} ({info['final_ess_ratio']:.1%})")
    print(f"  Step size:          {info['final_step_size']:.6f}")

    # Temperature progression
    temps = info['temperature_history']
    print(f"\nTemperature Schedule:")
    print(f"  Start:              beta = {temps[0]:.4f}")
    print(f"  End:                beta = {temps[-1]:.4f}")
    if len(temps) > 2:
        avg_step = (temps[-1] - temps[0]) / (len(temps) - 1)
        print(f"  Avg step size:      delta_beta = {avg_step:.4f}")

    # Acceptance rates
    accepts = info['accept_history']
    if accepts:
        avg_accept = sum(accepts) / len(accepts)
        print(f"\nAcceptance Rates:")
        print(f"  Average:            {avg_accept:.1%}", end="")
        if avg_accept < 0.2:
            print("  (low - consider smaller step)")
        elif avg_accept > 0.8:
            print("  (high - could use larger step)")
        else:
            print("  (good)")
        print(f"  Range:              [{min(accepts):.1%}, {max(accepts):.1%}]")

    # Timing info
    if "total_time" in info:
        print(f"\nTiming:")
        print(f"  Total SMC time:     {info['total_time']:.2f}s")
        if info['n_temperature_steps'] > 0:
            per_step = info['total_time'] / info['n_temperature_steps']
            print(f"  Per temp step:      {per_step:.3f}s")

    print("=" * 60)


def _bisect_next_temperature(
    particles: torch.Tensor,
    log_unnorm_prob: Callable,
    beta_current: float,
    target_ess_ratio: float = 0.9,
) -> float:
    """
    Adaptive temperature selection via bisection search.

    Finds the largest Δβ such that ESS(Δβ) ≥ target_ess_ratio × N, where N is
    the number of particles. This exploits the monotonicity of ESS in Δβ: larger
    temperature increments cause greater weight variance, thus lower ESS.

    The incremental importance weights for moving from β to β + Δβ are:

        w_i ∝ exp(Δβ × log π(x_i))

    where π(x) is the unnormalized target density. When Δβ is small, weights
    are nearly uniform (ESS ≈ N). When Δβ is large, weights become degenerate
    (ESS → 1). Bisection efficiently finds the threshold.

    Note:
        This implementation computes ESS from incremental weights only, not
        accounting for accumulated weights from previous steps. This is a common
        simplification that assumes particles are approximately equally weighted
        after each resampling step.

    References:
        - Jasra, A., Stephens, D. A., Doucet, A., & Tsagaris, T. (2011).
          Inference for Lévy-driven stochastic volatility models via adaptive
          sequential Monte Carlo. Scandinavian Journal of Statistics, 38(1), 1-22.
          (Original ESS-based adaptive temperature selection)

        - Schäfer, C., & Chopin, N. (2013). Sequential Monte Carlo on large
          binary sampling spaces. Statistics and Computing, 23(2), 163-184.
          (Bisection procedure with η* = 0.9, Procedure 2)

    Args:
        particles: Current particle positions, shape (N, d).
        log_unnorm_prob: Function returning log π(x) for the target distribution.
        beta_current: Current inverse temperature β ∈ [0, 1].
        target_ess_ratio: Target ESS as a fraction of N. Default 0.9 following
            Schäfer & Chopin (2013). Higher values give smaller, safer steps.

    Returns:
        Next inverse temperature β_next ∈ (β_current, 1].
    """
    with torch.no_grad():
        log_prob_values = log_unnorm_prob(particles)
    
    N = particles.shape[0]
    
    beta_low = beta_current
    beta_high = 1.0
    
    # Check if we can go directly to β=1
    delta_beta = beta_high - beta_current
    log_weights_test = delta_beta * log_prob_values
    ess_test = compute_ess_from_weights(log_weights_test)
    
    if ess_test / N >= target_ess_ratio:
        return 1.0
    
    # Binary search
    for _ in range(20):
        beta_mid = (beta_low + beta_high) / 2
        delta_beta = beta_mid - beta_current
        log_weights_test = delta_beta * log_prob_values
        ess_test = compute_ess_from_weights(log_weights_test)
        
        if ess_test / N > target_ess_ratio:
            beta_low = beta_mid
        else:
            beta_high = beta_mid
        
        if beta_high - beta_low < 1e-6:
            break
    
    return beta_low


def smc_reference_point_generator(
    energy_distribution: "BoltzmannEnergyDistribution",
    bounds: torch.Tensor,
    n_samples: int = 1000,
    n_temperatures: int = 20,
    n_mutation_steps: int = 10,
    adaptive_temperatures: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    stablize: bool = False,
    mutation_kernel: str = "rwm",
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    Generate reference points via Sequential Monte Carlo (SMC) sampling.

    This function samples from the Boltzmann distribution π(x) ∝ exp(-f(x)),
    where f(x) is the energy function provided by the energy_distribution.
    SMC uses temperature annealing to gradually move particles from the
    uniform prior to the target distribution, which is robust for multimodal targets.

    Temperature annealing explanation:
        We introduce a parameter β (inverse temperature) to create a sequence
        of intermediate distributions:

            π_β(x) ∝ exp(-β * f(x))

        - β = 0: exp(0) = 1 for all x, i.e., uniform distribution (easy to sample)
        - β = 1: exp(-f(x)), the target Boltzmann distribution

        By gradually increasing β from 0 to 1, particles smoothly transition
        from uniform coverage to concentrating in low-energy (high-probability)
        regions, avoiding the need to cross low-probability barriers.

    Algorithm overview:
        1. Initialize particles uniformly within bounds
        2. Anneal: gradually increase β from 0 to 1
        3. At each temperature step:
           - Reweight particles for the temperature change (importance weights)
           - Resample if effective sample size (ESS) drops too low
           - Apply RWM mutations to rejuvenate particle diversity
        4. Return final particles as reference points

    Args:
        energy_distribution: BoltzmannEnergyDistribution instance that provides
            compute_log_unnorm_density(X) for evaluating log π(x).
        bounds: (2, d) tensor of [lower_bounds, upper_bounds] defining the
            sampling domain.
        n_samples: Number of particles/samples to generate.
        n_temperatures: Number of temperature steps for fixed schedule.
            Used when adaptive_temperatures=False.
        n_mutation_steps: Number of RWM mutation steps per temperature level.
            More steps help particles explore but increase cost.
        adaptive_temperatures: If True, adaptively choose temperatures based
            on ESS. If False, use fixed linear schedule with n_temperatures steps.
            Default: False.
        seed: Optional random seed for full reproducibility.
        verbose: If True, print progress including ESS, acceptance rates,
            and resampling events.
        stablize: Whether to stabilize log density computations in
            energy_distribution (default: False).
        mutation_kernel: Mutation kernel; only "rwm" (Random Walk Metropolis) is supported.

    Returns:
        reference_points: (n_samples, d) tensor of samples from the target
            distribution, suitable for use as integration points in
            AB-SID-iVAR or similar acquisition functions.
    """
    device = bounds.device
    dtype = bounds.dtype
    d = bounds.shape[1]
    lb, ub = bounds[0], bounds[1]

    def log_prob(x: torch.Tensor) -> torch.Tensor:
        return energy_distribution.compute_log_unnorm_density(x, stabilize=stablize)

    particles, log_weights, info = sample_smc(
        log_unnorm_prob=log_prob,
        d=d,
        n_particles=n_samples,
        n_temperatures=n_temperatures,
        n_mutation_steps=n_mutation_steps,
        bounds=(lb, ub),
        adaptive_temperatures=adaptive_temperatures,
        device=device,
        dtype=dtype,
        seed=seed,
        verbose=verbose,
        mutation_kernel=mutation_kernel,
    )

    # Optional: resample according to final weights for unweighted samples
    final_ess_ratio = info["final_ess_ratio"]
    if final_ess_ratio < 0.9:
        # Use a derived seed for final resampling (if seed was provided)
        resample_generator = None
        if seed is not None:
            resample_generator = torch.Generator(device=device)
            resample_generator.manual_seed(seed + 1000000)  # Offset to avoid correlation
        particles = systematic_resample(particles, log_weights, generator=resample_generator)

    if return_info:
        return particles, info
    return particles


def uniform_reference_point_generator(
    bounds: torch.Tensor,
    n_samples: int = 1000,
    seed: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Generate uniform random reference points.

    Simple generator for IMSE/EPIG where uniform coverage is desired.
    Uses a manual seed for full reproducibility.

    Args:
        bounds: (2, d) tensor of [lower, upper] bounds
        n_samples: Number of points to generate
        seed: Random seed for reproducibility
        **kwargs: Ignored (for interface compatibility)

    Returns:
        (n_samples, d) tensor of uniform random samples scaled to bounds
    """
    d = bounds.shape[1]
    generator = torch.Generator(device=bounds.device)
    generator.manual_seed(seed)
    samples = torch.rand(n_samples, d, dtype=bounds.dtype, device=bounds.device, generator=generator)
    return bounds[0] + samples * (bounds[1] - bounds[0])