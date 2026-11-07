#!/usr/bin/env python
"""
CLI script to generate MCMC sensitivity analysis plots (SMC particle counts).

Usage:
    python fig_scripts/generate_mcmc_sensitivity_plot.py \
        --results_root exps/results/mcmc_sensitivity \
        --metric weighted_mse \
        --acq_types TS \
        --baselines RS \
        --suptitle "MCMC Sensitivity" \
        --output figures/mcmc_sensitivity.pdf

    # With custom dimensions and particle multipliers:
    python fig_scripts/generate_mcmc_sensitivity_plot.py \
        --results_root exps/results/mcmc_sensitivity \
        --dims 3 6 9 12 \
        --particle_mults 500 1000 1500 \
        --kernels rwm \
        --acq_types TS AB \
        --metric weighted_mse \
        --output figures/mcmc_sensitivity_full.pdf
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCMC sensitivity analysis plots (SMC particle counts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root folder containing MCMC sensitivity results",
    )

    # Optional arguments
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        help="Dimensions to plot (default: [3, 6, 9, 12])",
    )
    parser.add_argument(
        "--particle_mults",
        type=int,
        nargs="+",
        default=None,
        help="Particle multipliers (default: [500, 1000, 1500]). Actual particles = mult * dim",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        nargs="+",
        default=None,
        help="Kernels to compare (default: ['rwm'])",
    )
    parser.add_argument(
        "--acq_types",
        type=str,
        nargs="+",
        default=None,
        help="Acquisition types to plot (default: ['AB']). Options: TS, AB",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help="Baseline methods to include (e.g., RS for Random Sampling)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "weighted_mse"],
        help="Metric to plot (default: mse)",
    )
    parser.add_argument(
        "--figsize_per_panel",
        type=float,
        nargs=2,
        default=[5, 3.5],
        help="(width, height) per panel in inches (default: 5 3.5)",
    )
    parser.add_argument(
        "--suptitle",
        type=str,
        default=None,
        help="Overall figure title",
    )
    parser.add_argument(
        "--no_uncertainty",
        action="store_true",
        help="Hide uncertainty bands",
    )
    parser.add_argument(
        "--use_std",
        action="store_true",
        help="Use standard deviation instead of standard error",
    )
    parser.add_argument(
        "--n_markers",
        type=int,
        default=8,
        help="Number of markers per curve (default: 8)",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.0,
        help="Line width (default: 1.0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure (default: 300)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for figure (.png or .pdf). Shows plot if not provided.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively (in addition to saving)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    import matplotlib.pyplot as plt
    from sidal.benchmarking.visualization import (
        plot_mcmc_sensitivity,
    )

    # Generate figure
    fig = plot_mcmc_sensitivity(
        results_root=args.results_root,
        dims=args.dims,
        particle_mults=args.particle_mults,
        kernels=args.kernels,
        acq_types=args.acq_types,
        baselines=args.baselines,
        metric=args.metric,
        figsize_per_panel=tuple(args.figsize_per_panel),
        use_stderr=not args.use_std,
        show_fill=not args.no_uncertainty,
        n_markers=args.n_markers,
        linewidth=args.linewidth,
        suptitle=args.suptitle,
        save_path=None,  # We handle saving ourselves
        dpi=args.dpi,
    )

    # Save figure
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    # Show plot
    if args.show or not args.output:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
