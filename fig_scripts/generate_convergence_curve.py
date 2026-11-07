#!/usr/bin/env python
"""
CLI script to generate publication-ready convergence curves from benchmark results.

Usage:
python fig_scripts/generate_convergence_curve.py \
      --exp_root exps/results/formal_benchmark/continuous \
      --exp_names \
          Forrester_1d/matern52 \
          Gramacy_1d/matern52 \
          Branin_2d/matern52 \
          Gramacy_2d/matern52 \
          GP_prior_3d/matern52 \
          Hartmann_3d/matern52 \
          Ishigami_3d/matern52 \
          Hartmann_6d/matern52 \
      --methods RS US EPIG mu-SID-EPIG_NoC IMSE HAL TS-SID-iVAR_EWV AB-SID-iVAR_EWV \
      --metric weighted_mse \
      --max_cols 4 \
      --legend_ncol 8 \
      --figsize_per_panel 4 3 \
      --linewidth 1.5 \
      --markersize 5 \
      --output ./figures/benchmark_comparison.pdf

Ablation study figure:
  python fig_scripts/generate_convergence_curve.py \
      --exp_root exps/results/formal_benchmark/continuous \
      --exp_names \
          Forrester_1d/matern52 \
          Gramacy_1d/matern52 \
          Branin_2d/matern52 \
          Gramacy_2d/matern52 \
          GP_prior_3d/matern52 \
          Hartmann_3d/matern52 \
          Ishigami_3d/matern52 \
          Hartmann_6d/matern52 \
      --methods RS AB-SID-iVAR_EWV AB-SID-iVAR_NoC TS-SID-iVAR_EWV TS-SID-iVAR_NoC mu-SID-iVAR_EWV mu-SID-iVAR_NoC \
      --metric weighted_mse \
      --max_cols 4 \
      --legend_ncol 10 \
      --figsize_per_panel 4 3 \
      --linewidth 1.5 \
      --markersize 5 \
      --output ./figures/ablation_constraint.pdf
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready convergence curves from benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--exp_root",
        type=str,
        required=True,
        help="Root folder containing experiment result directories",
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment folder names (relative to exp_root)",
    )

    # Optional arguments
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Methods to plot (auto-detect if not provided)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to use (auto-detect if not provided)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="weighted_mse",
        choices=["mse", "weighted_mse", "nlml", "weighted_nlml"],
        help="Metric to plot (default: weighted_mse)",
    )
    parser.add_argument(
        "--max_cols",
        type=int,
        default=3,
        help="Maximum number of columns in the grid (default: 3)",
    )
    parser.add_argument(
        "--figsize_per_panel",
        type=float,
        nargs=2,
        default=[5, 4],
        help="(width, height) per panel in inches (default: 5 4)",
    )
    parser.add_argument(
        "--suptitle",
        type=str,
        default=None,
        help="Overall figure title",
    )
    parser.add_argument(
        "--exp_titles",
        type=str,
        nargs="+",
        default=None,
        help="Titles for each panel (uses exp_names if not provided)",
    )
    parser.add_argument(
        "--legend_loc",
        type=str,
        default="bottom",
        choices=["bottom", "right", "none"],
        help="Legend location (default: bottom)",
    )
    parser.add_argument(
        "--legend_ncol",
        type=int,
        default=None,
        help="Number of columns in legend (auto if not provided)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only show top k methods by final value",
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
        "--errorbar_freq",
        type=int,
        default=5,
        help="Frequency of error bars/markers (default: 5)",
    )
    parser.add_argument(
        "--n_markers",
        type=int,
        default=8,
        help="Number of markers per curve (default: 8)",
    )
    parser.add_argument(
        "--markersize",
        type=float,
        default=3.0,
        help="Size of markers (default: 3.0)",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.8,
        help="Line width (default: 0.8)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure (default: 300)",
    )
    parser.add_argument(
        "--max_iters_per_panel",
        type=int,
        nargs="+",
        default=None,
        help="Per-panel x-axis truncation (list of max iterations, one per exp).",
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
        generate_convergence_curve_for_paper,
    )

    # Generate figure
    fig = generate_convergence_curve_for_paper(
        exp_root=args.exp_root,
        exp_names=args.exp_names,
        methods=args.methods,
        seeds=args.seeds,
        metric=args.metric,
        max_cols=args.max_cols,
        figsize_per_panel=tuple(args.figsize_per_panel),
        use_stderr=not args.use_std,
        show_uncertainty=not args.no_uncertainty,
        show_fill=True,
        errorbar_freq=args.errorbar_freq,
        n_markers=args.n_markers,
        markersize=args.markersize,
        linewidth=args.linewidth,
        top_k=args.top_k,
        suptitle=args.suptitle,
        exp_titles=args.exp_titles,
        legend_loc=args.legend_loc,
        legend_ncol=args.legend_ncol,
        save_path=None,  # We handle saving ourselves
        dpi=args.dpi,
    )

    # Apply per-panel x-axis truncation if requested
    if args.max_iters_per_panel:
        if len(args.max_iters_per_panel) != len(args.exp_names):
            raise ValueError(
                f"--max_iters_per_panel has {len(args.max_iters_per_panel)} entries, "
                f"need {len(args.exp_names)} (one per exp)"
            )
        panel_axes = [ax for ax in fig.axes if ax.has_data()]
        for ax, xmax in zip(panel_axes, args.max_iters_per_panel):
            ax.set_xlim(0, xmax)

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
