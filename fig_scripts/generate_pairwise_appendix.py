#!/usr/bin/env python
"""
Appendix figure: Pairwise pred-vs-true scatter plots.

Layout: 4 rows × 4 columns
  Rows: Median 1 λ=25, Median 1 λ=75, Median 2 λ=25, Median 2 λ=75
  Cols: RS, US, IMSE, AB-SID-iVAR-X̄

Each panel shows true y (x-axis) vs predicted μ (y-axis) for all ~20k pool
points, colored by Boltzmann weight exp(λy), with R² annotation and y=x line.

Usage:
  python fig_scripts/generate_pairwise_appendix.py
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
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import gaussian_kde

from sidal.data_loaders import load_molecular_pool_data
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from botorch.models.transforms.outcome import Standardize

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Style setup (matching paper conventions)
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

# ---------------------------------------------------------------------------
# Method styling
# ---------------------------------------------------------------------------
METHOD_COLOR_MAP = {
    "RS": "#808080",
    "US": "#D2691E",
    "IMSE": "#228B22",
    "AB-SID-iVAR_EWV": "#B22222",
}
METHOD_LABEL_MAP = {
    "RS": "RS",
    "US": "US",
    "IMSE": "IMSE",
    "AB-SID-iVAR_EWV": r"AB-SID-iVAR-$\bar{\mathcal{X}}$",
}
ACQS = ["RS", "US", "IMSE", "AB-SID-iVAR_EWV"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("exps/pool")
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
# Rows: (dataset_key, lambda)
ROWS = [("med1", 25), ("med1", 75), ("med2", 25), ("med2", 75)]
SEED = 0
BATCH_SIZE = 2000


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


def predict_on_pool(model, pool_X):
    """Predict mean on full pool in batches."""
    mu_list = []
    with torch.no_grad():
        for i in range(0, len(pool_X), BATCH_SIZE):
            post = model.posterior(pool_X[i : i + BATCH_SIZE])
            mu_list.append(post.mean.squeeze(-1))
    return torch.cat(mu_list).numpy()


# ---------------------------------------------------------------------------
# Load pool data
# ---------------------------------------------------------------------------
print("Loading pool data...")
pool_data = {}
for ds_key, ds_info in DATASETS.items():
    pool_data[ds_key] = load_pool_data(ds_info["objective"])
    print(f"  {ds_info['label']}: {len(pool_data[ds_key][0])} points")

# ---------------------------------------------------------------------------
# Create figure: 4 rows × 4 columns + space for colorbars
# ---------------------------------------------------------------------------
n_rows, n_cols = 4, 4
panel_w, panel_h = 3.5, 3.5
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(panel_w * n_cols + 1.2, panel_h * n_rows),
    squeeze=False,
)

# Custom colormap: gray → tilted-P_λ purple (#7B68EE), matching composite figure
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "gray_purple", ["#BCBCBC", "#7B68EE"], N=256
)

# Store scatter mappables + norms per row (for colorbars added after layout)
row_mappables = {}
row_norms = {}
row_lams = {}

for row_idx, (ds_key, lam) in enumerate(ROWS):
    pool_X, pool_Y = pool_data[ds_key]
    y_true = pool_Y.numpy()

    # Compute log P_λ(y) = log p(y) + λy for coloring
    # Estimate base density p(y) via KDE
    kde = gaussian_kde(y_true)
    log_py = np.log(kde(y_true) + 1e-30)  # log p(y)
    log_plam = log_py + lam * y_true       # log P_λ(y) (unnormalized)
    log_plam_display = log_plam - log_plam.max()  # shift for display

    # Shared color normalization for this row
    vmin = np.percentile(log_plam_display, 1)  # clip extreme low tail
    vmax = 0.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    row_norms[row_idx] = norm
    row_lams[row_idx] = lam

    for col_idx, acq in enumerate(ACQS):
        ax = axes[row_idx, col_idx]
        rdir = result_dir_for(ds_key, lam)
        fpath = rdir / f"{acq}_seed{SEED}.pt"

        if not fpath.exists():
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray",
            )
            continue

        print(f"  {DATASETS[ds_key]['label']}, λ={lam}, {acq}...")
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        hp = data["model_hyperparameters"][-1]
        model = build_gp_from_saved(data["final_X"], data["final_Y"], hp)
        mu_pred = predict_on_pool(model, pool_X)
        del model

        # Identify queried pool indices by matching final_X against pool_X
        queried_X = data["final_X"]  # (n_queried, d)
        # Batched pairwise L1 distance to find exact matches
        queried_mask = np.zeros(len(pool_X), dtype=bool)
        for i in range(0, len(queried_X), 50):
            batch = queried_X[i : i + 50]  # (b, d)
            dists = torch.cdist(batch.float(), pool_X.float(), p=1)  # (b, N)
            min_idx = dists.argmin(dim=1)  # (b,)
            min_val = dists[torch.arange(len(batch)), min_idx]
            for idx, val in zip(min_idx, min_val):
                if val < 1e-6:
                    queried_mask[idx.item()] = True

        # Per-point sizes: larger for queried points
        sizes = np.where(queried_mask, 12, 3)

        # Sort by weight so high-weight points are drawn on top
        sort_order = np.argsort(log_plam_display)
        sc = ax.scatter(
            y_true[sort_order],
            mu_pred[sort_order],
            c=log_plam_display[sort_order],
            cmap=CMAP,
            norm=norm,
            s=sizes[sort_order],
            alpha=0.6,
            edgecolors="none",
            rasterized=True,
        )
        row_mappables[row_idx] = sc

        # y=x reference line
        lo = min(y_true.min(), mu_pred.min())
        hi = max(y_true.max(), mu_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            linewidth=1.0,
            alpha=0.5,
            zorder=5,
        )
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_aspect("equal", adjustable="box")

        # Column header (top row only)
        if row_idx == 0:
            ax.set_title(METHOD_LABEL_MAP[acq])

        # Axis labels — only show y-tick labels on the first column
        if col_idx == 0:
            ax.set_ylabel("Predicted $\\mu$")
        else:
            ax.set_yticklabels([])
        if row_idx == n_rows - 1:
            ax.set_xlabel("True $y$")

    # Row label on the left
    axes[row_idx, 0].annotate(
        f"{DATASETS[ds_key]['label']}\n$\\lambda={lam}$",
        xy=(-0.45, 0.5),
        xycoords="axes fraction",
        fontsize=15,
        ha="center",
        va="center",
        rotation=90,
    )

# ---------------------------------------------------------------------------
# Layout first, then add colorbars manually so they don't steal axis space
# ---------------------------------------------------------------------------
plt.tight_layout(w_pad=0.8, h_pad=1.0)
plt.subplots_adjust(left=0.1, right=0.89)

for row_idx in range(n_rows):
    if row_idx not in row_mappables:
        continue
    # Position colorbar to the right of the last column in this row
    # Get bounding box of rightmost axis (in figure coords)
    ax_right = axes[row_idx, -1]
    pos = ax_right.get_position()
    cbar_left = pos.x1 + 0.01
    cbar_bottom = pos.y0 + pos.height * 0.1
    cbar_width = 0.012
    cbar_height = pos.height * 0.8
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(row_mappables[row_idx], cax=cax)
    cbar.set_label(
        f"$\\log P_\\lambda(y)$", fontsize=11,
    )

output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "pairwise_appendix.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "pairwise_appendix.pdf", bbox_inches="tight")
print(f"Saved to {output_dir / 'pairwise_appendix.png'} and .pdf")
plt.close(fig)
