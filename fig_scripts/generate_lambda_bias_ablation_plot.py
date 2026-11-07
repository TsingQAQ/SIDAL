#!/usr/bin/env python
"""
Generate 4-row ablation plot for lambda-bias study.

Row 1: Contour of GP sample f(x) for b=0
Row 2: Convergence curves for b=0
Row 3: Contour of GP sample f(x) for b=GP
Row 4: Convergence curves for b=GP

Usage:
python fig_scripts/generate_lambda_bias_ablation_plot.py \
    --exp_root ./exps/results/lambda_bias_ablation \
    --methods RS US IMSE HAL AB-SID-iVAR_EWV TS-SID-iVAR_EWV \
    --output ./figures/lambda_bias_ablation.pdf
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_results(exp_dir: Path, method: str, seeds: Optional[List[int]] = None):
    """Load results for a method, handling various file naming patterns."""
    exp_dir = Path(exp_dir)

    # Method name mapping to handle file naming variations
    method_file_patterns = {
        "AB-SID-iVAR_EWV": "AB-SID-iVAR_EWV_zeroth_order_analytic",
        "AB-SID-iVAR_NoC": "AB-SID-iVAR_NoC_zeroth_order_analytic",
        "TS-SID-iVAR_EWV": "TS-SID-iVAR_EWV_thompson_sampling",
        "TS-SID-iVAR_NoC": "TS-SID-iVAR_NoC_thompson_sampling",
    }

    file_prefix = method_file_patterns.get(method, method)

    patterns = [
        f"{file_prefix}_seed*.pt",
        # Archived results carry an extra legacy "_beta0.0" tag
        f"{file_prefix}_beta0.0_seed*.pt",
        f"{method}_seed*.pt",
        f"{method}_*_seed*.pt",
    ]

    files = []
    for pattern in patterns:
        files = sorted(exp_dir.glob(pattern))
        if files:
            break

    if not files:
        return None, None

    results = []
    for f in files:
        match = re.search(r'seed(\d+)\.pt$', f.name)
        if match:
            seed = int(match.group(1))
            if seeds is None or seed in seeds:
                try:
                    data = torch.load(f, map_location='cpu', weights_only=False)
                    results.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {f}: {e}")

    if not results:
        return None, None

    return results, len(results)


def extract_metric(results: List[dict], metric: str = "weighted_mse"):
    """Extract metric curves from results."""
    curves = []
    for r in results:
        if 'metrics' in r and metric in r['metrics']:
            curves.append(r['metrics'][metric])
        elif 'metrics_history' in r and metric in r['metrics_history']:
            curves.append(r['metrics_history'][metric])
        elif 'history' in r and metric in r['history']:
            curves.append(r['history'][metric])

    if not curves:
        return None, None, None

    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    arr = np.array(curves)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    stderr = std / np.sqrt(len(curves))

    return mean, std, stderr


def create_test_function(problem_kwargs, device='cpu'):
    """Recreate test function from problem kwargs."""
    try:
        from sidal.test_functions import GPSampleWithBias
        func = GPSampleWithBias(**problem_kwargs)
        return func
    except Exception as e:
        print(f"Warning: Could not create test function: {e}")
        return None


def plot_function_contour(ax, exp_dir: Path, title: str, lam: float, seed: int = 0):
    """Plot contour of weighted density exp(λf(x))."""
    # Load one result to get problem_kwargs
    files = sorted(exp_dir.glob("RS_seed*.pt"))
    if not files:
        files = sorted(exp_dir.glob("*_seed0.pt"))

    if not files:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    data = torch.load(files[0], map_location='cpu', weights_only=False)
    problem_kwargs = data.get('problem_kwargs', {})

    # Recreate test function
    func = create_test_function(problem_kwargs)
    if func is None:
        ax.text(0.5, 0.5, "Cannot load function", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Create grid
    n_grid = 100
    x1 = np.linspace(0, 1, n_grid)
    x2 = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = torch.tensor(np.column_stack([X1.ravel(), X2.ravel()]), dtype=torch.float32)

    # Evaluate function
    with torch.no_grad():
        Y = func(X_grid).numpy()

    Z = Y.reshape(n_grid, n_grid)

    # Compute weighted density exp(λf)
    density = np.exp(lam * Z)
    density_norm = density / density.max()

    # Plot contour of weighted density (darker blue = higher density)
    contour = ax.contourf(X1, X2, density_norm, levels=50, cmap='Blues')
    ax.set_title(title)
    ax.set_aspect('equal')
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    return contour


def generate_lambda_bias_ablation_plot(
    exp_root: Union[str, Path],
    methods: List[str],
    metric: str = "weighted_mse",
    seeds: Optional[List[int]] = None,
    figsize_per_panel: tuple = (4.0, 2.8),
    linewidth: float = 1.5,
    markersize: float = 4,
    n_markers: int = 8,
    output_path: Optional[str] = None,
    dpi: int = 300,
    show_contour: bool = True,
):
    """
    Generate 4-row ablation plot for lambda-bias study.

    Layout (when show_contour=True):
        Row 1: Contour of exp(λf(x)) for b=0 (no bias)
        Row 2: Convergence curves for b=0
        Row 3: Contour of exp(λf(x)) for b~GP (GP bias)
        Row 4: Convergence curves for b~GP

    Uses nested GridSpec to group rows 1-2 and 3-4 tightly with a gap between groups.

    Args:
        exp_root: Root directory containing lambda_*_bias_* experiment folders.
        methods: List of method names to plot (e.g., ["RS", "US", "IMSE", "HAL"]).
        metric: Metric to plot for convergence curves (default: "weighted_mse").
        seeds: Optional list of seeds to include. If None, uses all available.
        figsize_per_panel: (width, height) per panel in inches.
        linewidth: Line width for convergence curves.
        markersize: Marker size for convergence curves.
        n_markers: Number of markers per curve.
        output_path: Path to save figure. If None, figure is not saved.
        dpi: Resolution for saved figure.
        show_contour: If True, show contour rows. If False, only convergence rows.

    Returns:
        matplotlib Figure object.
    """
    exp_root = Path(exp_root)

    lambdas = [
        (-0.5, "m0p5"),
        (-1.0, "m1p0"),
        (-2.0, "m2p0"),
        (-5.0, "m5p0"),
    ]

    bias_modes = [
        ("none", "$b(x)=0$"),
        ("gp", "$b(x) \\sim \\mathcal{GP}$"),
    ]

    n_cols = len(lambdas)
    n_rows = len(bias_modes) * (2 if show_contour else 1)

    # Color scheme - SAME as visualization.py
    method_color_map = {
        "RS": "#808080",              # neutral gray
        "US": "#D2691E",              # chocolate orange
        "IMSE": "#228B22",            # forest green
        "EPIG": "#17BECF",            # teal/cyan
        "HAL": "#9467BD",             # muted purple
        "AB-SID-iVAR_NoC": "#8B0000", # darker red
        "AB-SID-iVAR_EWV": "#B22222", # firebrick red
        "TS-SID-iVAR_NoC": "#00008B", # dark blue
        "TS-SID-iVAR_EWV": "#0080FF", # bright blue
        "mu-SID-iVAR_NoC": "#8B4513", # dark brown
        "mu-SID-iVAR_EWV": "#DEB887", # burlywood
        "mu-SID-EPIG_NoC": "#FF1493", # deep pink
    }

    # Label map for legend
    label_map = {
        "AB-SID-iVAR_EWV": r"AB-SID-iVAR-$\bar{\mathcal{X}}$",
        "AB-SID-iVAR_NoC": r"AB-SID-iVAR-$\phi$",
        "TS-SID-iVAR_EWV": r"TS-SID-iVAR-$\bar{\mathcal{X}}$",
        "TS-SID-iVAR_NoC": r"TS-SID-iVAR-$\phi$",
        "mu-SID-iVAR_EWV": r"$\mu$-SID-iVAR-$\bar{\mathcal{X}}$",
        "mu-SID-iVAR_NoC": r"$\mu$-SID-iVAR-$\phi$",
        "mu-SID-EPIG_NoC": r"$\mu$-SID-EPIG",
    }

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    # Set publication-ready style (matching visualization.py)
    plt.rcParams.update({
        'mathtext.fontset': 'dejavusans',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.linewidth': 1.0,
    })

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    # Use nested GridSpec: outer splits into 2 row groups, inner has tight spacing
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Outer GridSpec: 2 row groups with gap between them
    outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.28)

    # Inner GridSpecs: each has 2 rows (contour + curve) with tight spacing
    inner_gs_top = GridSpecFromSubplotSpec(2, n_cols, subplot_spec=outer_gs[0], hspace=0.05, wspace=0.25)
    inner_gs_bottom = GridSpecFromSubplotSpec(2, n_cols, subplot_spec=outer_gs[1], hspace=0.05, wspace=0.25)

    # Create axes array manually
    axes = np.empty((n_rows, n_cols), dtype=object)
    for j in range(n_cols):
        axes[0, j] = fig.add_subplot(inner_gs_top[0, j])  # Row 0: contour b=0
        axes[1, j] = fig.add_subplot(inner_gs_top[1, j])  # Row 1: curve b=0
        axes[2, j] = fig.add_subplot(inner_gs_bottom[0, j])  # Row 2: contour b~GP
        axes[3, j] = fig.add_subplot(inner_gs_bottom[1, j])  # Row 3: curve b~GP

    legend_handles = []
    legend_labels = []

    for bias_idx, (bias_mode, bias_label) in enumerate(bias_modes):
        for col_idx, (lam_val, lam_fmt) in enumerate(lambdas):
            exp_dir = exp_root / f"lambda_{lam_fmt}_bias_{bias_mode}"

            if show_contour:
                # Contour row
                contour_row = bias_idx * 2
                ax_contour = axes[contour_row, col_idx]

                title = f"$\\lambda={lam_val}$, {bias_label}"
                if exp_dir.exists():
                    plot_function_contour(ax_contour, exp_dir, title, lam=lam_val)
                else:
                    ax_contour.text(0.5, 0.5, "No data", ha='center', va='center',
                                   transform=ax_contour.transAxes)
                    ax_contour.set_title(title)

                # Remove aspect='equal' to allow flexible sizing
                ax_contour.set_aspect('auto')

                # Add ylabel for first column
                if col_idx == 0:
                    ax_contour.set_ylabel('Density')

                # Convergence row
                curve_row = bias_idx * 2 + 1
            else:
                curve_row = bias_idx

            ax = axes[curve_row, col_idx]

            if not exp_dir.exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"$\\lambda={lam_val}$")
                continue

            # Plot each method
            for m_idx, method in enumerate(methods):
                results, n_seeds_loaded = load_results(exp_dir, method, seeds)

                if results is None:
                    continue

                mean, std, stderr = extract_metric(results, metric)

                if mean is None:
                    continue

                color = method_color_map.get(method, f"C{m_idx}")
                label = label_map.get(method, method)
                marker = markers[m_idx % len(markers)]

                x = np.arange(len(mean))

                if len(x) > n_markers:
                    marker_indices = np.linspace(0, len(x)-1, n_markers, dtype=int)
                else:
                    marker_indices = np.arange(len(x))

                line, = ax.plot(x, mean, color=color, linewidth=linewidth, label=label)
                ax.plot(x[marker_indices], mean[marker_indices],
                       color=color, marker=marker, markersize=markersize, linestyle='none')
                ax.fill_between(x, mean - stderr, mean + stderr, color=color, alpha=0.2)

                # Collect legend from first convergence panel
                if bias_idx == 0 and col_idx == 0 and (not show_contour or curve_row == 1):
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_yscale('log')
            ax.set_xlabel('Iteration')
            if col_idx == 0:
                ax.set_ylabel('Weighted MSE')
            if not show_contour:
                ax.set_title(f"$\\lambda={lam_val}$, {bias_label}")

            # Add grid to convergence plots
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)


    # Legend styling matching visualization.py
    legend_kwargs = {
        'fontsize': 12,
        'frameon': True,
        'fancybox': True,
        'shadow': False,
        'framealpha': 0.95,
        'edgecolor': '#CCCCCC',
        'borderpad': 0.6,
        'columnspacing': 1.2,
        'handletextpad': 0.5,
        'handlelength': 1.8,
    }

    fig.legend(legend_handles, legend_labels,
               loc='lower center',
               ncol=len(methods),
               bbox_to_anchor=(0.5, 0.01),
               **legend_kwargs)

    # Align ylabels vertically across rows
    fig.align_ylabels(axes[:, 0])

    plt.subplots_adjust(bottom=0.10, left=0.06, right=0.98, top=0.96)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate lambda-bias ablation study plot")

    parser.add_argument("--exp_root", type=str, required=True,
                       help="Root directory containing lambda_*_bias_* folders")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["RS", "US", "IMSE", "HAL",
                               "AB-SID-iVAR_EWV", "TS-SID-iVAR_EWV"],
                       help="Methods to plot")
    parser.add_argument("--metric", type=str, default="weighted_mse")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--figsize_per_panel", type=float, nargs=2, default=[4.0, 2.8])
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--markersize", type=float, default=5)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-contour", action="store_true", help="Skip contour rows")

    args = parser.parse_args()

    fig = generate_lambda_bias_ablation_plot(
        exp_root=args.exp_root,
        methods=args.methods,
        metric=args.metric,
        seeds=args.seeds,
        figsize_per_panel=tuple(args.figsize_per_panel),
        linewidth=args.linewidth,
        markersize=args.markersize,
        output_path=args.output,
        dpi=args.dpi,
        show_contour=not args.no_contour,
    )

    if args.show or not args.output:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
