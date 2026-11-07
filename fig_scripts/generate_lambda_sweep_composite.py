#!/usr/bin/env python
"""
Composite figure for molecular pool lambda sweep experiments.

Layout: 3 rows × 4 columns
  Row 1: Boltzmann density histograms (MED1 λ=25, MED1 λ=75, MED2 λ=25, MED2 λ=75)
  Row 2: Weighted MSE convergence curves (median ± IQR over 10 seeds)
  Row 3: Enrichment curves — R² on top-k% pool by Boltzmann weight (seed 0)

Usage:
  python fig_scripts/generate_lambda_sweep_composite.py
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

from sidal.data_loaders import load_molecular_pool_data
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from botorch.models.transforms.outcome import Standardize

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Style setup (matching paper conventions from visualization.py)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.titleweight": "normal",
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.8,
        "lines.linewidth": 1.8,
    }
)

# Method styling (consistent with the paper)
METHOD_COLOR_MAP = {
    "RS": "#808080",
    "US": "#D2691E",
    "EPIG": "#17BECF",
    "IMSE": "#228B22",
    "AB-SID-iVAR_EWV": "#B22222",
}
METHOD_LABEL_MAP = {
    "RS": "RS",
    "US": "US",
    "EPIG": "EPIG",
    "IMSE": "IMSE",
    "AB-SID-iVAR_EWV": r"AB-SID-iVAR-$\bar{\mathcal{X}}$",
}
METHOD_MARKERS = {"RS": "o", "US": "s", "EPIG": "v", "IMSE": "^", "AB-SID-iVAR_EWV": "D"}
ACQS = ["RS", "US", "EPIG", "IMSE", "AB-SID-iVAR_EWV"]
N_MARKERS = 8
N_SEEDS = 10

LEGEND_KWARGS = {
    "fontsize": 12,
    "frameon": True,
    "fancybox": True,
    "shadow": False,
    "framealpha": 0.95,
    "edgecolor": "#CCCCCC",
    "borderpad": 0.6,
    "columnspacing": 1.2,
    "handletextpad": 0.4,
    "handlelength": 1.5,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("exps/results/formal_benchmark/pool")
DATASETS = {
    "med1": {
        "subdir": "molecular_med1_",
        "label": "Median 1",
        "objective": "med1",
    },
    "med2": {
        "subdir": "molecular_med2_",
        "label": "Median 2",
        "objective": "med2",
    },
}
LAMBDAS = [25, 75]
# All 3 rows share the same 4-panel layout: (dataset_key, lambda)
PANELS = [("med1", 25), ("med1", 75), ("med2", 25), ("med2", 75)]
# Enrichment curve k-values (%)
ENRICH_KS = np.array([1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100])


def load_pool_data(objective):
    config = {
        "problem": {
            "type": "pool",
            "dataset_path": "exps/data/guacamol_train_data_first_20k_dedup.csv",
            "objective": objective,
            "fingerprint_type": "morgan",
            "fingerprint_bits": 2048,
            "fingerprint_radius": 2,
        },
        "data_split": {"test_size": 0, "test_split_seed": 42},
    }
    pool_X, pool_Y, _, _ = load_molecular_pool_data(config, verbose=False)
    return pool_X, pool_Y


def build_gp_from_saved(train_X, train_Y, hyperparams):
    """Build GP and load saved hyperparameters (no refitting)."""
    kernel = ScaleKernel(TanimotoKernel())
    model = SingleTaskGP(
        train_X,
        train_Y.unsqueeze(-1) if train_Y.dim() == 1 else train_Y,
        covar_module=kernel,
        outcome_transform=Standardize(m=1),
    )
    # Set saved hyperparameters
    model.covar_module.outputscale = torch.tensor(hyperparams["outputscale"])
    model.mean_module.constant.data.fill_(hyperparams["mean_constant"])
    model.likelihood.noise_covar.noise = torch.tensor(hyperparams["noise"])
    model.likelihood.noise_covar.raw_noise.requires_grad_(False)
    model.eval()
    return model


def result_dir_for(ds_key, lam):
    return (
        BASE_DIR / f"l{lam}" / "pool" / DATASETS[ds_key]["subdir"] / "tanimoto"
    )


def load_convergence_curves(ds_key, lam, acq):
    """Load weighted_mse from all seeds, return (n_seeds, n_iters) array."""
    rdir = result_dir_for(ds_key, lam)
    curves = []
    for seed in range(N_SEEDS):
        fpath = rdir / f"{acq}_seed{seed}.pt"
        if not fpath.exists():
            continue
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        curves.append(np.array(data["metrics"]["weighted_mse"]))
    if not curves:
        return None
    # Truncate to minimum length across seeds
    min_len = min(len(c) for c in curves)
    return np.array([c[:min_len] for c in curves])


# ---------------------------------------------------------------------------
# Load pool data
# ---------------------------------------------------------------------------
print("Loading pool data...")
pool_data = {}
for ds_key, ds_info in DATASETS.items():
    pool_data[ds_key] = load_pool_data(ds_info["objective"])
    print(f"  {ds_info['label']}: {len(pool_data[ds_key][0])} points")

# ---------------------------------------------------------------------------
# Create figure: 3 rows × 4 columns
# Match benchmark_comparison.png: figsize_per_panel=(4,3), tight_layout
# ---------------------------------------------------------------------------
panel_w, panel_h = 4, 3
n_rows, n_cols = 3, 4
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(panel_w * n_cols, panel_h * n_rows + 0.8),
                         squeeze=False)

# ========================== ROW 1: Boltzmann density ==========================
print("Plotting Row 1: Boltzmann density histograms...")
for col, (ds_key, lam) in enumerate(PANELS):
    ax = axes[0, col]
    _, pool_Y = pool_data[ds_key]
    y = pool_Y.numpy()

    # Boltzmann weights
    log_w = lam * y
    log_w_stab = log_w - log_w.max()
    w = np.exp(log_w_stab)
    w_norm = w / w.sum()

    # Pool Y histogram (background)
    n_pool, bin_edges, _ = ax.hist(y, bins=80, color="gray", alpha=0.3,
                                    density=False, zorder=1)
    pool_peak = n_pool.max()

    # Boltzmann-shifted histogram: multiply each bin count by exp(λ · y_bin_center)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    log_shift = lam * bin_centers
    log_shift -= log_shift.max()  # stabilize
    boltz_counts = n_pool * np.exp(log_shift)
    # Scale so peak matches pool histogram peak
    boltz_peak = boltz_counts.max()
    scale = pool_peak / boltz_peak if boltz_peak > 0 else 1.0
    boltz_counts_scaled = boltz_counts * scale
    # Plot as bar chart matching the histogram bins
    bin_width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, boltz_counts_scaled, width=bin_width,
           color="#7B68EE", alpha=0.55, zorder=2)

    ax.set_title(f"{DATASETS[ds_key]['label']} ($\\lambda={lam}$)")
    ax.set_yticks([])
    ax.set_xlabel("Function value $y$")

# ========================== ROW 2: Convergence curves =========================
print("Plotting Row 2: Convergence curves (median ± IQR, 10 seeds)...")
convergence_handles = {}
for col, (ds_key, lam) in enumerate(PANELS):
    ax = axes[1, col]

    for acq in ACQS:
        curves = load_convergence_curves(ds_key, lam, acq)
        if curves is None:
            continue
        color = METHOD_COLOR_MAP[acq]
        label = METHOD_LABEL_MAP[acq]
        marker = METHOD_MARKERS[acq]

        median_curve = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25, axis=0)
        q75 = np.percentile(curves, 75, axis=0)
        iters = np.arange(len(median_curve))

        ax.plot(iters, median_curve, color=color, linewidth=1.5)
        ax.fill_between(iters, q25, q75, color=color, alpha=0.2)
        marker_indices = np.linspace(0, len(iters) - 1, N_MARKERS, dtype=int)
        ax.plot(iters[marker_indices], median_curve[marker_indices],
                marker=marker, color=color, markersize=6, linestyle="None")

        if label not in convergence_handles:
            convergence_handles[label] = Line2D(
                [0], [0], color=color, marker=marker, markersize=6,
                linewidth=1.8, label=label,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    if col == 0:
        ax.set_ylabel("Weighted MSE")
    ax.set_axisbelow(True)

# ======================== ROW 3: Enrichment curves ============================
print("Plotting Row 3: Enrichment curves (R² vs top-k%, median ± IQR over seeds)...")
for col, (ds_key, lam) in enumerate(PANELS):
    ax = axes[2, col]
    pool_X, pool_Y = pool_data[ds_key]
    y_true = pool_Y.numpy()

    # Boltzmann weights
    log_w = lam * y_true
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_norm = w / w.sum()
    sort_idx = np.argsort(-w_norm)  # descending by weight

    rdir = result_dir_for(ds_key, lam)
    for acq in ACQS:
        # Collect enrichment curves across seeds
        all_r2_curves = []
        for seed in range(N_SEEDS):
            fpath = rdir / f"{acq}_seed{seed}.pt"
            if not fpath.exists():
                continue
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            hp = data["model_hyperparameters"][-1]
            model = build_gp_from_saved(data["final_X"], data["final_Y"], hp)

            mu_list = []
            with torch.no_grad():
                for i in range(0, len(pool_X), 2000):
                    post = model.posterior(pool_X[i:i + 2000])
                    mu_list.append(post.mean.squeeze(-1))
            mu_pred = torch.cat(mu_list).numpy()

            r2_at_k = []
            for k in ENRICH_KS:
                n_top = max(1, int(len(y_true) * k / 100))
                idx = sort_idx[:n_top]
                yt, yp = y_true[idx], mu_pred[idx]
                ss_res = np.sum((yt - yp) ** 2)
                ss_tot = np.sum((yt - yt.mean()) ** 2)
                r2_at_k.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
            all_r2_curves.append(r2_at_k)
            del model

        if not all_r2_curves:
            continue
        all_r2 = np.array(all_r2_curves)  # (n_seeds, n_ks)
        median_r2 = np.median(all_r2, axis=0)
        q25 = np.percentile(all_r2, 25, axis=0)
        q75 = np.percentile(all_r2, 75, axis=0)

        color = METHOD_COLOR_MAP[acq]
        marker = METHOD_MARKERS[acq]
        ax.plot(ENRICH_KS, median_r2, color=color, linewidth=1.8)
        ax.fill_between(ENRICH_KS, q25, q75, color=color, alpha=0.2)
        marker_indices = np.linspace(0, len(ENRICH_KS) - 1, N_MARKERS, dtype=int)
        ax.plot(ENRICH_KS[marker_indices], median_r2[marker_indices],
                marker=marker, color=color, markersize=6, linestyle="None")

        print(f"  {DATASETS[ds_key]['label']}, λ={lam}, {acq}: "
              f"{len(all_r2_curves)} seeds")

    ax.set_xlabel("Top $k$%")
    if col == 0:
        ax.set_ylabel("$R^2$ on top $k$%")
    ax.set_xscale("log")
    ax.set_xlim(110, 0.8)  # flip: 100% (left) → 1% (right), log scale
    ax.set_xticks([100, 20, 5, 1])
    ax.set_xticklabels(["100", "20", "5", "1"])
    ax.minorticks_off()
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

# ---------------------------------------------------------------------------
# Single bottom legend (density items + method items)
# ---------------------------------------------------------------------------
density_handles = [
    Patch(facecolor="gray", alpha=0.3, label="Pool $y$"),
    Patch(facecolor="#7B68EE", alpha=0.55, label=r"Tilted $P_\lambda(y)$"),
]
method_handles = [
    Line2D([0], [0], color=METHOD_COLOR_MAP[acq], marker=METHOD_MARKERS[acq],
           markersize=6, linewidth=1.8, label=METHOD_LABEL_MAP[acq])
    for acq in ACQS
]
all_handles = density_handles + method_handles
fig.legend(
    handles=all_handles, loc="lower center",
    ncol=len(all_handles), fontsize=12,
    bbox_to_anchor=(0.5, -0.02),
    frameon=True, fancybox=True, framealpha=0.95, edgecolor="#CCCCCC",
    handletextpad=0.4, columnspacing=1.2, handlelength=1.5,
    borderpad=0.6,
)

plt.tight_layout(h_pad=1.0, w_pad=0.5)
plt.subplots_adjust(bottom=0.08, hspace=0.25)

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "composite_lambda_sweep.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "composite_lambda_sweep.pdf", bbox_inches="tight")
print(f"Saved to {output_dir / 'composite_lambda_sweep.png'} and .pdf")
plt.close(fig)
