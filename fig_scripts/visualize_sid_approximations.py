"""
Visualize SID approximation variants on 2D GP prior.

Compares acquisition function contours for:
1. MC-SID (MC=32): "Ground truth" via averaging over 32 posterior samples
2. AB-SID (0th order): exp(λμ + λ²/2 σ²) - closed form, biased
3. AB-SID (1st order): Adds covariance correction (Eq. 27 in paper)
4. TS-SID (MC=1): Single posterior sample exp(λf̃) - unbiased, high variance

Usage:
    python scripts/visualize_sid_approximations.py
    python scripts/visualize_sid_approximations.py --seed 42 --n_train 5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths


def create_gp_prior_sample(
    dim: int = 2,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    seed: int = 0,
    n_grid: int = 50,
):
    """Create a sample from GP prior as ground truth function."""
    torch.manual_seed(seed)

    # Create grid
    x = torch.linspace(0, 1, n_grid)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    # Create kernel
    base_kernel = MaternKernel(nu=2.5)
    base_kernel.lengthscale = lengthscale
    kernel = ScaleKernel(base_kernel)
    kernel.outputscale = outputscale

    # Sample from prior (use double precision for numerical stability)
    grid = grid.double()
    K = kernel(grid, grid).to_dense().double()
    K = K + 1e-6 * torch.eye(K.shape[0], dtype=torch.float64)
    L = torch.linalg.cholesky(K)
    z = torch.randn(K.shape[0], dtype=torch.float64)
    f = L @ z

    return grid.float(), f.reshape(n_grid, n_grid).float(), kernel


def create_model_with_fixed_hyperparams(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    noise: float = 1e-4,
):
    """Create GP model with fixed hyperparameters (oracle setting)."""
    # Use double precision
    train_X = train_X.double()
    train_Y = train_Y.double()

    # Create custom kernel with Matern 5/2
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]))

    model = SingleTaskGP(
        train_X,
        train_Y.unsqueeze(-1),
        covar_module=covar_module,
    )

    # Fix hyperparameters
    model.covar_module.base_kernel.lengthscale = lengthscale
    model.covar_module.outputscale = outputscale
    model.likelihood.noise = noise

    return model


def compute_mc_sid_weights(
    model: SingleTaskGP,
    grid: torch.Tensor,
    n_samples: int = 32,
    lambda_val: float = -1.0,
    seed: int = 0,
    num_rff_features: int = 2048,
):
    """
    Compute MC estimate of E[P_f(x)] by averaging over posterior samples.

    This is the "ground truth" for comparison.

    For n_samples > 1:
        Each sample f_i gives normalized distribution P_i(x) = exp(λf_i(x)) / Z_i
        We average: P_MC(x) = (1/N) Σ_i P_i(x)

        Note: We must normalize each sample BEFORE averaging, because samples with
        different partition functions Z_i should contribute equally to the average.

    For n_samples == 1:
        Returns unnormalized log weights λf (same as TS-SID), since normalization
        happens later in the acquisition function computation.
    """
    torch.manual_seed(seed)

    log_weights_list = []

    for i in range(n_samples):
        # Draw a posterior sample using Matheron paths
        prior_sampler = partial(draw_kernel_feature_paths, num_features=num_rff_features)
        sample_path = draw_matheron_paths(
            model, sample_shape=torch.Size([1]), prior_sampler=prior_sampler
        )

        with torch.no_grad():
            f_sample = sample_path(grid).squeeze()  # (n_grid,)

        log_weights = lambda_val * f_sample
        log_weights_list.append(log_weights)

    log_weights_stack = torch.stack(log_weights_list, dim=0)  # (n_samples, n_grid)

    if n_samples == 1:
        # For single sample, return unnormalized log weights (exactly same as before)
        # Normalization happens later in compute_acquisition_surface
        return log_weights_stack.squeeze(0)
    else:
        # For multiple samples, normalize each sample BEFORE averaging
        # P_i(x) = exp(λf_i(x)) / Z_i where Z_i = Σ_x' exp(λf_i(x'))
        log_Z = torch.logsumexp(log_weights_stack, dim=1, keepdim=True)  # (n_samples, 1)
        P_normalized = torch.exp(log_weights_stack - log_Z)  # (n_samples, n_grid)

        # Average normalized distributions: P_MC(x) = (1/N) Σ_i P_i(x)
        P_avg = P_normalized.mean(dim=0)  # (n_grid,)

        # Return log for consistency with other functions
        # Use clamp to avoid log(0) for numerical stability
        log_mean_weights = torch.log(torch.clamp(P_avg, min=1e-10))

        return log_mean_weights


def compute_ab_sid_zeroth_order(
    model: SingleTaskGP,
    grid: torch.Tensor,
    lambda_val: float = -1.0,
):
    """
    Compute AB-SID zeroth order approximation.

    E[exp(λf)] = exp(λμ + λ²σ²/2)
    log E[exp(λf)] = λμ + λ²σ²/2
    """
    with torch.no_grad():
        posterior = model.posterior(grid)
        mu = posterior.mean.squeeze()
        sigma_sq = posterior.variance.squeeze()

    log_weights = lambda_val * mu + (lambda_val ** 2) * sigma_sq / 2
    return log_weights


def compute_ab_sid_first_order(
    model: SingleTaskGP,
    grid: torch.Tensor,
    lambda_val: float = -1.0,
):
    """
    Compute AB-SID first order approximation with covariance correction.

    From paper Eq. 27:
    E[P_f(x*)] ≈ E[A]/E[B] · (1 - E_{P̃}[exp(λ²k) - 1])

    For unnormalized density:
    log p̃(x*) = log E[A] + log(1 - E_{P̃}[exp(λ²k) - 1])
    """
    with torch.no_grad():
        posterior = model.posterior(grid)
        mu = posterior.mean.squeeze()
        sigma_sq = posterior.variance.squeeze()
        K = posterior.distribution.covariance_matrix  # (n, n)

    # Zeroth order term
    log_T0 = lambda_val * mu + (lambda_val ** 2) * sigma_sq / 2

    # Compute P̃ (normalized zeroth-order weights)
    log_P_tilde_unnorm = log_T0.clone()
    log_Z = torch.logsumexp(log_P_tilde_unnorm, dim=0)
    P_tilde = torch.exp(log_P_tilde_unnorm - log_Z)  # (n,)

    # Compute exp(λ²k) - 1
    lambda_sq = lambda_val ** 2
    exp_lambda_sq_k = torch.exp(lambda_sq * K) - 1  # (n, n)

    # Correction term: E_{P̃}[exp(λ²k) - 1] for each point
    correction_term = exp_lambda_sq_k @ P_tilde  # (n,)

    # Full correction: 1 - correction_term
    correction = 1.0 - correction_term

    # Check validity: first-order Taylor is only valid when correction > 0
    n_invalid = (correction <= 0).sum().item()
    if n_invalid > 0:
        print(f"  WARNING: First-order correction invalid at {n_invalid}/{len(correction)} points")
        print(f"           (correction_term > 1, i.e., λ²·variance too large)")

    correction_clamped = torch.clamp(correction, min=1e-10)

    log_weights = log_T0 + torch.log(correction_clamped)
    return log_weights


def compute_ts_sid(
    model: SingleTaskGP,
    grid: torch.Tensor,
    lambda_val: float = -1.0,
    seed: int = 0,
    num_rff_features: int = 2048,
):
    """
    Compute TS-SID (Thompson Sampling): single posterior sample.

    log p̃(x) = λf̃(x) where f̃ ~ P(f|D)
    """
    torch.manual_seed(seed)

    prior_sampler = partial(draw_kernel_feature_paths, num_features=num_rff_features)
    sample_path = draw_matheron_paths(
        model, sample_shape=torch.Size([1]), prior_sampler=prior_sampler
    )

    with torch.no_grad():
        f_sample = sample_path(grid).squeeze()

    log_weights = lambda_val * f_sample
    return log_weights


def compute_weighted_fantasy_variance(
    model: SingleTaskGP,
    grid: torch.Tensor,
    candidate: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Compute E_P[σ²(x | candidate)] = Σ P(x_i) σ²(x_i | candidate)

    Uses Schur complement for efficient fantasy variance computation.
    """
    # Get current posterior variance
    with torch.no_grad():
        posterior = model.posterior(grid)
        var_before = posterior.variance.squeeze()

    # Compute fantasy variance after observing candidate
    train_X = model.train_inputs[0]
    covar_module = model.covar_module
    noise_var = model.likelihood.noise.item()

    # Kernel matrices
    K_grid_train = covar_module(grid, train_X).to_dense()
    K_train_train = covar_module(train_X, train_X).to_dense()
    K_train_train_noisy = K_train_train + noise_var * torch.eye(train_X.shape[0])

    K_grid_cand = covar_module(grid, candidate.unsqueeze(0)).to_dense().squeeze(-1)
    K_cand_train = covar_module(candidate.unsqueeze(0), train_X).to_dense().squeeze(0)
    K_cand_cand = covar_module(candidate.unsqueeze(0), candidate.unsqueeze(0)).to_dense().item()

    # Cholesky for training data
    L_train = torch.linalg.cholesky(K_train_train_noisy)

    # Current variance reduction from training
    Ainv_Kgt = torch.cholesky_solve(K_grid_train.t(), L_train)
    red_train = (K_grid_train * Ainv_Kgt.t()).sum(dim=1)

    # Schur complement for candidate
    Ainv_Kct = torch.cholesky_solve(K_cand_train.unsqueeze(-1), L_train).squeeze(-1)
    S = K_cand_cand + noise_var - (K_cand_train * Ainv_Kct).sum()

    # Additional variance reduction from candidate
    K_grid_train_Ainv_Kct = K_grid_train @ Ainv_Kct
    numerator = (K_grid_cand - K_grid_train_Ainv_Kct) ** 2
    additional_red = numerator / S

    # Fantasy variance
    prior_var = covar_module(grid, grid).diagonal()
    var_after = prior_var - red_train - additional_red
    var_after = torch.clamp(var_after, min=1e-10)

    # Weighted expectation
    weighted_var = (weights * var_after).sum()

    return weighted_var, var_after


def compute_acquisition_surface(
    model: SingleTaskGP,
    eval_grid: torch.Tensor,
    reference_grid: torch.Tensor,
    log_weights: torch.Tensor,
    n_eval: int,
):
    """Compute acquisition function values over evaluation grid."""
    # Normalize weights
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))

    acq_values = torch.zeros(n_eval * n_eval)

    for i, candidate in enumerate(eval_grid):
        weighted_var, _ = compute_weighted_fantasy_variance(
            model, reference_grid, candidate, weights
        )
        acq_values[i] = -weighted_var  # Negative because we minimize variance

    return acq_values.reshape(n_eval, n_eval)


def setup_publication_style():
    """Set up publication-ready plot style with smaller figures and bigger text."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'dejavusans',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
    })


def main(
    seed: int = 42,
    n_train: int = 5,
    lengthscale: float = 0.2,
    outputscale: float = 1.0,
    lambda_val: float = -1.0,
    n_grid: int = 30,
    n_mc: int = 1024,
    save_path: str = None,
):
    """Main visualization function."""
    setup_publication_style()
    print(f"Creating visualization with seed={seed}, n_train={n_train}")

    # Create ground truth function from GP prior
    grid, f_true, _ = create_gp_prior_sample(
        dim=2, lengthscale=lengthscale, outputscale=outputscale,
        seed=seed, n_grid=n_grid
    )

    # Sample training points
    torch.manual_seed(seed + 100)
    train_indices = torch.randperm(n_grid * n_grid)[:n_train]
    train_X = grid[train_indices]
    train_Y = f_true.flatten()[train_indices]

    print(f"Training points: {n_train}")
    print(f"Grid size: {n_grid}x{n_grid}")

    # Create model with fixed hyperparameters
    model = create_model_with_fixed_hyperparams(
        train_X, train_Y,
        lengthscale=lengthscale,
        outputscale=outputscale,
    )
    model.eval()

    # Compute different SID approximations
    print("Computing MC-SID (MC=32)...")
    log_weights_mc = compute_mc_sid_weights(
        model, grid, n_samples=n_mc, lambda_val=lambda_val, seed=seed
    )

    print("Computing AB-SID (0th order)...")
    log_weights_ab0 = compute_ab_sid_zeroth_order(model, grid, lambda_val=lambda_val)

    print("Computing AB-SID (1st order)...")
    log_weights_ab1 = compute_ab_sid_first_order(model, grid, lambda_val=lambda_val)

    print("Computing TS-SID (MC=1)...")
    log_weights_ts = compute_ts_sid(model, grid, lambda_val=lambda_val, seed=seed)

    # Compute acquisition surfaces
    print("Computing acquisition surfaces...")
    n_eval = 25  # Coarser grid for acquisition evaluation (speed)
    x_eval = torch.linspace(0, 1, n_eval)
    xx_eval, yy_eval = torch.meshgrid(x_eval, x_eval, indexing='ij')
    eval_grid = torch.stack([xx_eval.flatten(), yy_eval.flatten()], dim=-1)

    acq_mc = compute_acquisition_surface(model, eval_grid, grid, log_weights_mc, n_eval)
    acq_ab0 = compute_acquisition_surface(model, eval_grid, grid, log_weights_ab0, n_eval)
    acq_ab1 = compute_acquisition_surface(model, eval_grid, grid, log_weights_ab1, n_eval)
    acq_ts = compute_acquisition_surface(model, eval_grid, grid, log_weights_ts, n_eval)

    # Create figure - smaller panels with bigger text
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))

    titles = [
        f'MC-SID (MC={n_mc})\n"Ground Truth"',
        'AB-SID (0th order)\nexp(λμ + λ²σ²/2)',
        'AB-SID (1st order)\n+ covariance correction',
        'TS-SID (MC=1)\nexp(λf̃)',
    ]
    acq_surfaces = [acq_mc, acq_ab0, acq_ab1, acq_ts]

    # Find global min/max for consistent colorscale
    all_acq = torch.stack(acq_surfaces)
    vmin, vmax = all_acq.min().item(), all_acq.max().item()

    for ax, acq, title in zip(axes, acq_surfaces, titles):
        # Plot acquisition contour
        contour = ax.contourf(
            xx_eval.detach().numpy(), yy_eval.detach().numpy(), acq.detach().numpy(),
            levels=20, cmap='viridis', vmin=vmin, vmax=vmax
        )

        # Add contour lines
        ax.contour(
            xx_eval.detach().numpy(), yy_eval.detach().numpy(), acq.detach().numpy(),
            levels=10, colors='white', alpha=0.3, linewidths=0.5
        )

        # Plot training points
        ax.scatter(
            train_X[:, 0].detach().numpy(), train_X[:, 1].detach().numpy(),
            c='red', s=80, marker='x', linewidths=2, label='Train'
        )

        # Find and mark optimal point
        opt_idx = acq.argmax()
        opt_x = opt_idx // n_eval
        opt_y = opt_idx % n_eval
        ax.scatter(
            x_eval[opt_x].item(), x_eval[opt_y].item(),
            c='yellow', s=150, marker='*', edgecolors='black', linewidths=1,
            label='Next query'
        )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axes, label='Acquisition Value', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)

    # Add legend to first plot
    axes[0].legend(loc='upper right', fontsize=8)

    plt.suptitle(
        f'SID Acquisition Function Comparison (λ={lambda_val}, ℓ={lengthscale}, σ²={outputscale}, n_train={n_train})',
        fontsize=11, y=1.02
    )
    plt.tight_layout(w_pad=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.savefig('sid_approximation_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved to sid_approximation_comparison.png")

    plt.show()

    # Print some statistics
    print("\n=== Weight Distribution Statistics ===")
    for name, log_w in [("MC-SID", log_weights_mc), ("AB-0", log_weights_ab0),
                         ("AB-1", log_weights_ab1), ("TS-SID", log_weights_ts)]:
        w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
        print(f"{name}: max_weight={w.max():.4f}, entropy={-(w * torch.log(w + 1e-10)).sum():.2f}")


def main_two_rows(
    seed: int = 42,
    n_train_few: int = 3,
    n_train_many: int = 10,
    lengthscale: float = 0.3,
    outputscale: float = 1.0,
    lambda_val: float = -1.0,
    n_grid: int = 30,
    n_mc: int = 1024,
    save_path: str = None,
):
    """
    Create 2-row visualization comparing few vs many training points.

    Row 1: Few training points (high uncertainty)
    Row 2: Many training points (lower uncertainty)

    Columns: MC-SID, AB-0, AB-1, TS-SID
    """
    setup_publication_style()
    print(f"Creating 2-row visualization: few={n_train_few}, many={n_train_many}")

    # Create ground truth function from GP prior
    grid, f_true, _ = create_gp_prior_sample(
        dim=2, lengthscale=lengthscale, outputscale=outputscale,
        seed=seed, n_grid=n_grid
    )

    # Smaller figure with bigger text
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))

    # Use same grid for both reference points and plotting
    x_eval = torch.linspace(0, 1, n_grid)
    xx_eval, yy_eval = torch.meshgrid(x_eval, x_eval, indexing='ij')

    all_acq_surfaces = []  # For global colorscale

    for row_idx, n_train in enumerate([n_train_few, n_train_many]):
        print(f"\n=== Row {row_idx + 1}: n_train={n_train} ===")

        # Sample training points uniformly (not from grid)
        torch.manual_seed(seed + 100 + row_idx * 1000)
        train_X = torch.rand(n_train, 2)
        # Interpolate f_true at train_X using nearest grid point
        grid_idx_x = (train_X[:, 0] * (n_grid - 1)).round().long().clamp(0, n_grid - 1)
        grid_idx_y = (train_X[:, 1] * (n_grid - 1)).round().long().clamp(0, n_grid - 1)
        train_Y = f_true[grid_idx_x, grid_idx_y]

        # Create model
        model = create_model_with_fixed_hyperparams(
            train_X, train_Y,
            lengthscale=lengthscale,
            outputscale=outputscale,
        )
        model.eval()

        # Compute SID approximations
        print("  Computing MC-SID...")
        log_weights_mc = compute_mc_sid_weights(
            model, grid, n_samples=n_mc, lambda_val=lambda_val, seed=seed
        )
        print("  Computing AB-0...")
        log_weights_ab0 = compute_ab_sid_zeroth_order(model, grid, lambda_val=lambda_val)
        print("  Computing AB-1...")
        log_weights_ab1 = compute_ab_sid_first_order(model, grid, lambda_val=lambda_val)
        print("  Computing TS-SID...")
        log_weights_ts = compute_ts_sid(model, grid, lambda_val=lambda_val, seed=seed)

        # Compute acquisition surfaces
        print("  Computing acquisition surfaces...")
        acq_mc = compute_acquisition_surface(model, grid, grid, log_weights_mc, n_grid)
        acq_ab0 = compute_acquisition_surface(model, grid, grid, log_weights_ab0, n_grid)
        acq_ab1 = compute_acquisition_surface(model, grid, grid, log_weights_ab1, n_grid)
        acq_ts = compute_acquisition_surface(model, grid, grid, log_weights_ts, n_grid)

        acq_surfaces = [acq_mc, acq_ab0, acq_ab1, acq_ts]
        titles = [
            f'MC-SID (MC={n_mc})',
            'AB-SID (0th order)',
            'AB-SID (1st order)',
            'TS-SID (MC=1)',
        ]

        all_acq_surfaces.extend(acq_surfaces)

        # Plot this row
        for col_idx, (acq, title) in enumerate(zip(acq_surfaces, titles)):
            ax = axes[row_idx, col_idx]

            contour = ax.contourf(
                xx_eval.detach().numpy(), yy_eval.detach().numpy(), acq.detach().numpy(),
                levels=20, cmap='viridis'
            )
            ax.contour(
                xx_eval.detach().numpy(), yy_eval.detach().numpy(), acq.detach().numpy(),
                levels=6, colors='white', alpha=0.7, linewidths=0.8,
                linestyles='solid', negative_linestyles='solid'
            )

            # Training points (red dots)
            ax.scatter(
                train_X[:, 0].detach().numpy(), train_X[:, 1].detach().numpy(),
                c='red', s=40, marker='o', edgecolors='white', linewidths=0.5, zorder=10
            )

            # Optimal point
            opt_idx = acq.argmax()
            opt_x = opt_idx // n_grid
            opt_y = opt_idx % n_grid
            ax.scatter(
                x_eval[opt_x].item(), x_eval[opt_y].item(),
                c='yellow', s=120, marker='*', edgecolors='black', linewidths=1, zorder=11
            )

            if row_idx == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

        # Print stats for this row
        print(f"  Weight stats:")
        for name, log_w in [("MC-SID", log_weights_mc), ("AB-0", log_weights_ab0),
                             ("AB-1", log_weights_ab1), ("TS-SID", log_weights_ts)]:
            w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
            print(f"    {name}: entropy={-(w * torch.log(w + 1e-10)).sum():.2f}")

    plt.tight_layout(h_pad=0.5, w_pad=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.savefig('sid_two_rows_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved to sid_two_rows_comparison.png")

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=5,
                        help="For single-row mode")
    parser.add_argument("--n_train_few", type=int, default=3,
                        help="Few points for 2-row mode")
    parser.add_argument("--n_train_many", type=int, default=10,
                        help="Many points for 2-row mode")
    parser.add_argument("--lengthscale", type=float, default=0.3)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--lambda_val", type=float, default=-1.0)
    parser.add_argument("--n_grid", type=int, default=30)
    parser.add_argument("--n_mc", type=int, default=1024)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--two_rows", action="store_true",
                        help="Create 2-row comparison (few vs many points)")
    args = parser.parse_args()

    if args.two_rows:
        main_two_rows(
            seed=args.seed,
            n_train_few=args.n_train_few,
            n_train_many=args.n_train_many,
            lengthscale=args.lengthscale,
            outputscale=args.outputscale,
            lambda_val=args.lambda_val,
            n_grid=args.n_grid,
            n_mc=args.n_mc,
            save_path=args.save_path,
        )
    else:
        main(
            seed=args.seed,
            n_train=args.n_train,
            lengthscale=args.lengthscale,
            outputscale=args.outputscale,
            lambda_val=args.lambda_val,
            n_grid=args.n_grid,
            n_mc=args.n_mc,
            save_path=args.save_path,
        )
