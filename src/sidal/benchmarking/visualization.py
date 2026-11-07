#!/usr/bin/env python3
"""Visualization utilities for active learning benchmarking results."""

import torch
from pathlib import Path
from typing import Union


def _find_result_file(exp_dir: Path, method: str, seed: int) -> Path:
    """
    Find the result file for a method, tolerating extra filename tags.

    Tries the exact name ``{method}_seed{seed}.pt`` first, then the pattern
    ``{method}_*seed{seed}.pt``. The pattern covers the tags the runner appends
    (e.g. ``AB-SID-iVAR_EWV`` -> ``AB-SID-iVAR_EWV_zeroth_order_analytic_seed0.pt``)
    as well as archived results that additionally carry a legacy ``_beta0.0`` tag
    (``AB-SID-iVAR_EWV_zeroth_order_analytic_beta0.0_seed0.pt``).

    Args:
        exp_dir: Experiment directory
        method: Method name or filename prefix (e.g., "AB-SID-iVAR_EWV")
        seed: Seed index

    Returns:
        Path to the result file

    Raises:
        FileNotFoundError: If no matching file is found
    """
    # If method already is the full filename prefix, use directly
    direct_path = exp_dir / f"{method}_seed{seed}.pt"
    if direct_path.exists():
        return direct_path

    # Pattern matching for names with extra tags
    pattern = f"{method}_*seed{seed}.pt"
    matches = list(exp_dir.glob(pattern))

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple matches - prefer one without the legacy beta tag
        for m in matches:
            if "beta" not in m.name:
                return m
        # If all carry the legacy tag, return first match but warn
        print(f"Warning: Multiple files match '{method}' with seed {seed}: {[m.name for m in matches]}")
        return matches[0]
    else:
        raise FileNotFoundError(
            f"No result file found for method='{method}', seed={seed} in {exp_dir}. "
            f"Tried: {direct_path.name} and pattern {pattern}"
        )


def plot_convergence(
    exp_dir: Union[str, Path],
    methods: Union[str, list],
    seeds: Union[int, list, None] = None,
    title: str = None,
    metric: str = "mse",
    use_stderr: bool = True,
    show_uncertainty: bool = True,
    show_fill: bool = True,
    errorbar_freq: int = 5,
    n_markers: int = None,
    markersize: float = 3.0,
    figsize: tuple = (8, 6),
    linewidth: float = 1.0,
    top_k: int = None,
    ax=None,
    show_legend: bool = True,
    colors: list = None,
    markers: list = None,
    linestyles: list = None,
):
    """
    Plot convergence for multiple methods with mean ± std/stderr.

    Args:
        exp_dir: Path to experiment directory (e.g., "results_gp_prior/")
        methods: Single method or list of methods to compare (e.g., ["RS", "IMSE", "AB-SID-iVAR"])
        seeds: Single seed, list of seeds, or None to auto-detect from files
        title: Plot title
        metric: Which metric to plot ("mse", "weighted_mse", "nlml", "weighted_nlml")
        use_stderr: If True, show standard error; if False, show standard deviation
        show_uncertainty: If True, show uncertainty (fill or error bars); if False, show only mean lines
        show_fill: If True, show shaded region; if False, show error bars (only used if show_uncertainty=True)
        errorbar_freq: Frequency of error bars (every N iterations). Ignored if n_markers is set.
        n_markers: Number of markers per curve. If set, overrides errorbar_freq to ensure
            consistent number of markers regardless of iteration count.
        markersize: Size of markers (default: 3.0)
        figsize: Figure size tuple (only used if ax is None)
        linewidth: Line width for plot lines (default: 2.0)
        top_k: If specified, only show the top k methods with lowest final metric value (default: None = show all)
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.
        show_legend: Whether to show legend (default: True). Set False for multi-panel figures.
        colors: Optional list of colors for methods. Auto-generated if None.
        markers: Optional list of markers for methods. Auto-generated if None.
        linestyles: Optional list of linestyles for methods. Auto-generated if None.

    Returns:
        If ax is None: matplotlib Figure object
        If ax is provided: dict of {method_name: line_handle} for creating shared legends

    Example:
        >>> from sidal.benchmarking.visualization import plot_convergence
        >>> # Single method, single seed
        >>> fig = plot_convergence("results/", "AB-SID-iVAR", seeds=0)
        >>> # Multiple methods, multiple seeds (shows mean ± stderr)
        >>> fig = plot_convergence("results/", ["RS", "IMSE", "AB-SID-iVAR"], seeds=[0, 1, 2, 3, 4])
        >>> # Auto-detect seeds from files
        >>> fig = plot_convergence("results/", ["RS", "IMSE"], seeds=None)
        >>> # Show only top 5 methods
        >>> fig = plot_convergence("results/", methods, top_k=5)
        >>> # Plot on existing axes (for multi-panel figures)
        >>> fig, axes = plt.subplots(1, 3)
        >>> plot_convergence("results/exp1", methods, ax=axes[0], show_legend=False, title="Exp 1")
        >>> plot_convergence("results/exp2", methods, ax=axes[1], show_legend=False, title="Exp 2")
        >>> lines = plot_convergence("results/exp3", methods, ax=axes[2], show_legend=False, title="Exp 3")
        >>> fig.legend(lines.values(), lines.keys(), loc='lower center', ncol=4)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re

    exp_dir = Path(exp_dir)

    # Handle single method or list
    if isinstance(methods, str):
        methods = [methods]

    # Handle single seed or list of seeds, or auto-detect
    if seeds is None:
        # Auto-detect seeds from files in exp_dir
        seed_pattern = re.compile(r"_seed(\d+)\.pt$")
        found_seeds = set()
        for f in exp_dir.glob("*_seed*.pt"):
            match = seed_pattern.search(f.name)
            if match:
                found_seeds.add(int(match.group(1)))
        seeds = sorted(found_seeds) if found_seeds else [0]
    elif isinstance(seeds, int):
        seeds = [seeds]

    # First pass: load all data for each method
    method_data = {}
    actual_seeds_loaded = set()

    for method in methods:
        all_metric = []
        for seed in seeds:
            try:
                result_path = _find_result_file(exp_dir, method, seed)
            except FileNotFoundError:
                print(f"Warning: No file found for {method} seed {seed}, skipping")
                continue
            history = torch.load(result_path, weights_only=False)
            if metric in history["metrics"]:
                all_metric.append(history["metrics"][metric])
                actual_seeds_loaded.add(seed)

        if len(all_metric) == 0:
            print(f"Warning: No data for {method}, skipping")
            continue

        all_metric = np.array(all_metric)  # shape: (n_seeds, n_iterations)
        mean_metric = all_metric.mean(axis=0)
        final_value = mean_metric[-1]  # Final iteration value for ranking

        method_data[method] = {
            'all_metric': all_metric,
            'mean_metric': mean_metric,
            'final_value': final_value,
        }

    # Filter to top_k methods if specified
    if top_k is not None and top_k < len(method_data):
        # Sort by final value (ascending for metrics like mse where lower is better)
        sorted_methods = sorted(method_data.keys(), key=lambda m: method_data[m]['final_value'])
        top_methods = sorted_methods[:top_k]
        method_data = {m: method_data[m] for m in top_methods}
        print(f"Showing top {top_k} methods: {top_methods}")

    # Generate unique colors for each method using a colormap (if not provided)
    n_methods = len(method_data)
    import matplotlib.cm as cm

    if colors is None:
        if n_methods <= 10:
            cmap = cm.get_cmap('tab10')
            colors = [cmap(i) for i in range(n_methods)]
        elif n_methods <= 20:
            cmap = cm.get_cmap('tab20')
            colors = [cmap(i) for i in range(n_methods)]
        else:
            cmap = cm.get_cmap('hsv')
            colors = [cmap(i / n_methods) for i in range(n_methods)]

    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>', '8', 'H']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']

    # Create figure only if ax not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot each method and collect line handles
    # Build a mapping from method name to original index for consistent colors
    method_to_orig_idx = {m: i for i, m in enumerate(methods)} if methods else {}

    line_handles = {}
    for idx, method in enumerate(method_data.keys()):
        # Use original method index for color/marker/linestyle to ensure consistency
        orig_idx = method_to_orig_idx.get(method, idx)
        color = colors[orig_idx % len(colors)]
        marker = markers[orig_idx % len(markers)]
        linestyle = linestyles[orig_idx % len(linestyles)]

        all_metric = method_data[method]['all_metric']
        iterations = np.arange(0, all_metric.shape[1])
        n_points = len(iterations)

        # Calculate marker frequency: use n_markers if set, else errorbar_freq
        if n_markers is not None and n_markers > 0:
            mark_every = max(1, n_points // n_markers)
        else:
            mark_every = errorbar_freq

        if len(all_metric) == 1:
            line, = ax.plot(iterations, all_metric[0], color=color, linewidth=linewidth,
                           linestyle=linestyle, marker=marker, markevery=mark_every,
                           markersize=markersize, label=method)
        else:
            mean_metric = all_metric.mean(axis=0)
            median_metric = np.median(all_metric, axis=0)
            # Use percentiles for log-scale friendly uncertainty bands
            lower_pct = np.percentile(all_metric, 25, axis=0)  # 25th percentile
            upper_pct = np.percentile(all_metric, 75, axis=0)  # 75th percentile

            std_metric = all_metric.std(axis=0)
            if use_stderr:
                std_metric = std_metric / np.sqrt(len(all_metric))

            # Plot median for log-scale (more robust), mean otherwise
            central_metric = median_metric  # Use median - more robust for log scale
            line, = ax.plot(iterations, central_metric, color=color, linewidth=linewidth,
                           linestyle=linestyle, marker=marker, markevery=mark_every,
                           markersize=markersize, label=method)

            if show_uncertainty:
                if show_fill:
                    # Use percentiles (IQR) for log-scale friendly bands
                    ax.fill_between(iterations, lower_pct, upper_pct,
                                    alpha=0.2, color=color,
                                    edgecolor=color, linewidth=0.5)
                else:
                    errorbar_idx = iterations[::errorbar_freq]
                    ax.errorbar(errorbar_idx, central_metric[::errorbar_freq],
                                yerr=[central_metric[::errorbar_freq] - lower_pct[::errorbar_freq],
                                      upper_pct[::errorbar_freq] - central_metric[::errorbar_freq]],
                                color=color, fmt='none', capsize=3, capthick=1.5)

        line_handles[method] = line

    # Finalize plot
    ax.set_xlabel('Iteration')
    # Properly capitalize metric labels (MSE, NLML should be uppercase)
    metric_label = metric.replace('_', ' ').title().replace('Mse', 'MSE').replace('Nlml', 'NLML')
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
    ax.set_yscale('log')

    # Set reasonable number of y-ticks for log scale (aim for 4-6 ticks, every ~2 orders of magnitude)
    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # numticks controls max ticks; matplotlib will space them reasonably
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    # Legend handling
    if show_legend:
        if len(method_data) > 4:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        else:
            ax.legend(loc='best', fontsize=10)

    # Title handling
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif fig is not None:  # Only auto-generate title if we created the figure
        err_type = "stderr" if use_stderr else "std"
        n_seeds_loaded = len(actual_seeds_loaded)
        auto_title = f'{metric_label} Convergence ({n_seeds_loaded} seeds, ±{err_type})'
        ax.set_title(auto_title, fontsize=14, fontweight='bold')

    # Return figure if we created it, otherwise return line handles for shared legend
    if fig is not None:
        plt.tight_layout()
        return fig
    else:
        return line_handles


def generate_convergence_curve_for_paper(
    exp_root: Union[str, Path],
    exp_names: list,
    methods: Union[str, list] = None,
    seeds: Union[int, list, None] = None,
    metric: str = "weighted_mse",
    max_cols: int = 3,
    figsize_per_panel: tuple = (5, 4),
    use_stderr: bool = True,
    show_uncertainty: bool = True,
    show_fill: bool = True,
    errorbar_freq: int = 5,
    n_markers: int = 8,
    markersize: float = 3.0,
    linewidth: float = 1.0,
    top_k: int = None,
    suptitle: str = None,
    exp_titles: list = None,
    legend_loc: str = "bottom",
    legend_ncol: int = None,
    save_path: Union[str, Path] = None,
    dpi: int = 300,
):
    """
    Generate publication-ready multi-panel convergence curves.

    Creates a grid of convergence plots, one for each experiment, with a
    shared legend to avoid redundancy. Designed for paper figures.

    Args:
        exp_root: Root folder containing experiment result directories
        exp_names: List of experiment folder names (relative to exp_root)
        methods: Methods to plot (auto-detect from first experiment if None)
        seeds: Seeds to use (None = auto-detect)
        metric: Metric to plot ("mse", "weighted_mse", "nlml", "weighted_nlml")
        max_cols: Maximum number of columns in the grid
        figsize_per_panel: (width, height) per panel in inches
        use_stderr: Use standard error (True) or std (False)
        show_uncertainty: Show uncertainty bands
        show_fill: Use shaded fill (True) or error bars (False)
        errorbar_freq: Frequency of error bars/markers (ignored if n_markers is set)
        n_markers: Number of markers per curve (default: 8). Ensures consistent markers.
        markersize: Size of markers (default: 3.0)
        linewidth: Line width
        top_k: Only show top k methods by final value
        suptitle: Overall figure title
        exp_titles: List of titles for each panel (uses exp_names if None)
        legend_loc: Legend location ("bottom", "right", or "none")
        legend_ncol: Number of columns in legend (auto if None)
        save_path: Path to save figure (None = don't save)
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object

    Example:
        >>> from sidal.benchmarking.visualization import (
        ...     generate_convergence_curve_for_paper
        ... )
        >>> fig = generate_convergence_curve_for_paper(
        ...     exp_root="./exps/results/formal_benchmark",
        ...     exp_names=["Branin_2d/matern52", "Hartmann_3d/matern52", "Hartmann_6d/matern52"],
        ...     methods=["RS", "US", "IMSE", "AB-SID-iVAR"],
        ...     max_cols=3,
        ...     metric="weighted_mse",
        ...     suptitle="Active Learning Convergence Comparison",
        ...     exp_titles=["Branin 2D", "Hartmann 3D", "Hartmann 6D"],
        ... )
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import seaborn as sns
    import re

    # Set modern, clean style optimized for publication (ICML figure*)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': False,  # Use matplotlib's mathtext (no LaTeX installation needed)
        'mathtext.fontset': 'dejavusans',  # Math font that matches sans-serif
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.titleweight': 'normal',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.alpha': 0.6,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.0,
    })

    exp_root = Path(exp_root)
    n_exps = len(exp_names)

    # Calculate grid layout
    n_cols = min(max_cols, n_exps)
    n_rows = (n_exps + n_cols - 1) // n_cols

    # Auto-detect methods from first experiment if not provided
    if methods is None:
        first_exp_dir = exp_root / exp_names[0]
        method_pattern = re.compile(r"^(.+?)_seed\d+\.pt$")
        found_methods = set()
        for f in first_exp_dir.glob("*_seed*.pt"):
            match = method_pattern.match(f.name)
            if match:
                found_methods.add(match.group(1))
        methods = sorted(found_methods)
        print(f"Auto-detected methods: {methods}")

    if isinstance(methods, str):
        methods = [methods]

    # Set up titles - format as "Name ($d=X$)"
    if exp_titles is None:
        exp_titles = []
        for name in exp_names:
            # Extract dimension and function name from patterns like "Forrester_1d/matern52"
            base = name.replace("/matern52", "")
            # Match patterns like "1d_Forrester" or "Forrester_1d"
            import re
            match = re.match(r"(\d+)d_(.+)", base)  # e.g., "1d_Forrester"
            if match:
                dim, func_name = match.groups()
                func_name = func_name.replace("_", " ")
                exp_titles.append(f"{func_name} ($d={dim}$)")
            else:
                match = re.match(r"(.+)_(\d+)d", base)  # e.g., "Forrester_1d"
                if match:
                    func_name, dim = match.groups()
                    func_name = func_name.replace("_", " ")
                    exp_titles.append(f"{func_name} ($d={dim}$)")
                else:
                    # Fallback
                    exp_titles.append(base.replace("_", " "))

    # Generate consistent colors across all panels
    # Color scheme: NoC (φ, dashed) is darker, EWV (X̄, solid) is lighter in same family
    method_color_map = {
        "RS": "#808080",              # neutral gray - baseline
        "US": "#D2691E",              # muted chocolate orange
        "IMSE": "#228B22",            # forest green (muted)
        "EPIG": "#17BECF",            # teal/cyan (distinct from red)
        "HAL": "#9467BD",             # muted purple
        "AB-SID-iVAR_NoC": "#8B0000", # darker red (φ, dashed)
        "AB-SID-iVAR_EWV": "#B22222", # red!70!black - firebrick (X̄, solid)
        "TS-SID-iVAR_NoC": "#00008B", # dark blue (φ, dashed)
        "TS-SID-iVAR_EWV": "#0080FF", # BRIGHT blue (X̄, solid)
        "mu-SID-iVAR_NoC": "#8B4513", # dark brown (φ, dashed)
        "mu-SID-iVAR_EWV": "#DEB887", # light brown/burlywood (X̄, solid)
        "mu-SID-EPIG_NoC": "#17BECF", # teal/cyan (matches EPIG family)
        "TS-SID-iVAR": "#8C564B",     # brown
        "AB-SID-iVAR": "#17BECF",     # cyan
    }

    n_methods = len(methods)
    colors = []
    for method in methods:
        if method in method_color_map:
            colors.append(method_color_map[method])
        else:
            # Fallback to Tableau colors for unknown methods
            fallback_colors = sns.color_palette("tab10", n_methods)
            colors.append(fallback_colors[len(colors) % len(fallback_colors)])

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P', '<', '>', '8', 'H']
    # Use dashed lines only for NoC variants, solid for everything else
    linestyles = ['--' if 'NoC' in method else '-' for method in methods]

    # Calculate figure size
    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows
    if legend_loc == "bottom":
        fig_height += 0.8
    elif legend_loc == "right":
        fig_width += 2.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Plot each experiment
    all_lines = {}
    for idx, exp_name in enumerate(exp_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        exp_dir = exp_root / exp_name
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} does not exist, skipping")
            ax.set_visible(False)
            continue

        lines = plot_convergence(
            exp_dir=exp_dir,
            methods=methods,
            seeds=seeds,
            metric=metric,
            use_stderr=use_stderr,
            show_uncertainty=show_uncertainty,
            show_fill=show_fill,
            errorbar_freq=errorbar_freq,
            n_markers=n_markers,
            markersize=markersize,
            linewidth=linewidth,
            top_k=top_k,
            ax=ax,
            show_legend=False,
            title=exp_titles[idx] if idx < len(exp_titles) else None,
            colors=colors,
            markers=markers,
            linestyles=linestyles,
        )
        all_lines.update(lines)

        # Only show ylabel on leftmost column
        if col != 0:
            ax.set_ylabel('')

    # Hide unused axes
    for idx in range(n_exps, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Add shared legend with modern styling
    if legend_loc != "none" and all_lines:
        handles = list(all_lines.values())
        labels = list(all_lines.keys())

        # Pretty labels for legend (LaTeX formatting)
        label_map = {
            "HAL": "GHAL",
            "AB-SID-iVAR_NoC": r"AB-SID-iVAR-$\phi$",
            "AB-SID-iVAR_EWV": r"AB-SID-iVAR-$\bar{\mathcal{X}}$",
            "TS-SID-iVAR_NoC": r"TS-SID-iVAR-$\phi$",
            "TS-SID-iVAR_EWV": r"TS-SID-iVAR-$\bar{\mathcal{X}}$",
            "mu-SID-iVAR_NoC": r"$\mu$-SID-iVAR-$\phi$",
            "mu-SID-iVAR_EWV": r"$\mu$-SID-iVAR-$\bar{\mathcal{X}}$",
            "mu-SID-EPIG_NoC": r"$\mu$-SID-EPIG",
        }
        labels = [label_map.get(l, l) for l in labels]
        if legend_ncol is None:
            legend_ncol = min(len(labels), 5)

        legend_kwargs = {
            'fontsize': 12,
            'frameon': True,
            'fancybox': True,
            'shadow': False,
            'framealpha': 0.95,
            'edgecolor': '#CCCCCC',
            'borderpad': 0.6,
            'columnspacing': 1.2,
            'handletextpad': 0.4,
            'handlelength': 1.5,
        }

        if legend_loc == "bottom":
            fig.legend(handles, labels, loc='lower center',
                       bbox_to_anchor=(0.5, -0.08), ncol=legend_ncol,
                       **legend_kwargs)
        elif legend_loc == "right":
            fig.legend(handles, labels, loc='center left',
                       bbox_to_anchor=(1.02, 0.5), ncol=1,
                       **legend_kwargs)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=15, fontweight='normal', y=0.98)

    plt.tight_layout(h_pad=2.5, w_pad=0.5)  # More vertical padding between rows
    if legend_loc == "bottom":
        plt.subplots_adjust(bottom=0.18, hspace=0.35)  # More room for legend

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def plot_mcmc_sensitivity(
    results_root: Union[str, Path],
    dims: list = None,
    particle_mults: list = None,
    kernels: list = None,
    acq_types: list = None,
    baselines: list = None,
    metric: str = "mse",
    figsize_per_panel: tuple = (5, 3.5),
    use_stderr: bool = True,
    show_fill: bool = True,
    n_markers: int = 8,
    linewidth: float = 1.0,
    suptitle: str = None,
    save_path: Union[str, Path] = None,
    dpi: int = 300,
):
    """
    Plot MCMC sensitivity analysis: 2 rows with 4 panels each.

    Row 1: Convergence curves (MSE vs iteration)
    Row 2: Bar chart of SMC timing per kernel

    Args:
        results_root: Root folder containing MCMC sensitivity results
            (e.g., "exps/results/mcmc_sensitivity")
        dims: List of dimensions to plot (default: [3, 6, 9, 12])
        particle_mults: Particle multipliers (default: [500, 1000, 1500])
            Actual particles = mult * dim
        kernels: Kernels to compare (default: ["rwm"])
        acq_types: Acquisition types to plot (default: ["AB"])
            Options: "TS" (Thompson Sampling), "AB" (Analytic Boltzmann)
            Directory suffix will be _TS or _AB
        baselines: Baseline methods to include (default: None)
            Options: "RS" (Random Sampling)
            Plotted as dashed gray lines for reference
        metric: Metric to plot ("mse" or "weighted_mse")
        figsize_per_panel: (width, height) per panel
        use_stderr: Use standard error (True) or std (False)
        show_fill: Use shaded uncertainty region
        n_markers: Number of markers per curve
        linewidth: Line width
        suptitle: Overall figure title
        save_path: Path to save figure
        dpi: DPI for saved figure

    Returns:
        matplotlib Figure object

    Example:
        >>> from sidal.benchmarking.visualization import plot_mcmc_sensitivity
        >>> fig = plot_mcmc_sensitivity(
        ...     results_root="exps/results/mcmc_sensitivity",
        ...     acq_types=["AB", "TS"],  # Compare both acquisition types
        ...     baselines=["RS"],  # Include random sampling baseline
        ...     suptitle="MCMC Sensitivity: AB vs TS",
        ... )
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.lines import Line2D

    # Publication-quality style (matches benchmark_comparison.png)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'dejavusans',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.titleweight': 'normal',
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.3,
        'grid.alpha': 0.6,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.6,
    })

    results_root = Path(results_root)

    # Defaults
    if dims is None:
        dims = [3, 6, 9, 12]
    if particle_mults is None:
        particle_mults = [500, 1000, 1500]
    if kernels is None:
        kernels = ["rwm"]
    if acq_types is None:
        acq_types = ["AB"]
    if baselines is None:
        baselines = []

    n_dims = len(dims)

    # Create figure with 2 rows, n_dims columns + space for legend at bottom
    fig_width = figsize_per_panel[0] * n_dims
    fig_height = figsize_per_panel[1] * 2 + 0.8
    fig, axes = plt.subplots(2, n_dims, figsize=(fig_width, fig_height), squeeze=False)

    # Colors by acquisition function (aligned with benchmark_comparison.png)
    # AB uses reds, TS uses blues. Particle count -> shade within the family.
    # Use dark->light so smallest particle count is darkest (easier to read).
    ab_colors = ["#8B0000", "#B22222", "#CD5C5C", "#E9967A"]  # darkred -> lightcoral/salmon
    ts_colors = ["#00008B", "#1E90FF", "#4FC3F7", "#87CEEB"]  # darkblue -> lightblue
    rs_color = "#808080"  # neutral gray, matching benchmark_comparison.png

    def particle_color(acq_type, idx):
        palette = ab_colors if acq_type == "AB" else ts_colors
        return palette[idx % len(palette)]

    # Acquisition type -> marker (consistent across particle counts)
    acq_markers = {"AB": "D", "TS": "o"}  # D=diamond (AB), o=circle (TS)
    acq_linestyles = {"AB": "-", "TS": "-"}  # both solid; distinguished by color family

    all_lines = {}

    # Store timing data for row 2
    timing_data = {dim: {pmult: {kernel: {acq: [] for acq in acq_types} for kernel in kernels} for pmult in particle_mults} for dim in dims}

    # Row 1: Plot convergence curves for each dimension
    for dim_idx, dim in enumerate(dims):
        ax = axes[0, dim_idx]

        for pmult_idx, pmult in enumerate(particle_mults):
            n_particles = pmult * dim

            for kernel in kernels:
                for acq_type in acq_types:
                    # Build experiment directory name with suffix
                    exp_name = f"d{dim}_p{n_particles}_{kernel}_{acq_type}"
                    exp_dir = results_root / exp_name

                    if not exp_dir.exists():
                        print(f"Warning: {exp_dir} not found, skipping")
                        continue

                    color = particle_color(acq_type, pmult_idx)
                    pmult_label = str(pmult)
                    acq_display = r"AB-$\bar{\mathcal{X}}$" if acq_type == "AB" else r"TS-$\bar{\mathcal{X}}$"
                    if len(acq_types) > 1:
                        label = f"{acq_display}, {pmult_label}·d"
                    else:
                        label = f"{pmult_label}·d particles"

                    # Find and load result files
                    result_files = list(exp_dir.glob("*_seed*.pt"))
                    if not result_files:
                        print(f"Warning: No results in {exp_dir}")
                        continue

                    all_metric = []
                    all_smc_times = []
                    for rf in result_files:
                        try:
                            history = torch.load(rf, weights_only=False)
                            if metric in history.get("metrics", {}):
                                all_metric.append(history["metrics"][metric])
                            # Collect SMC timing
                            smc_times = history.get("timing", {}).get("smc_sampling", [])
                            if smc_times:
                                all_smc_times.extend([t for t in smc_times if t > 0])
                        except Exception as e:
                            print(f"Warning: Failed to load {rf}: {e}")

                    # Store timing for row 2
                    if all_smc_times:
                        timing_data[dim][pmult][kernel][acq_type] = all_smc_times

                    if not all_metric:
                        continue

                    # Compute mean and stderr
                    all_metric = np.array(all_metric)
                    mean_metric = all_metric.mean(axis=0)
                    n_seeds = len(all_metric)
                    if use_stderr and n_seeds > 1:
                        uncertainty = all_metric.std(axis=0) / np.sqrt(n_seeds)
                    else:
                        uncertainty = all_metric.std(axis=0)

                    iterations = np.arange(len(mean_metric))

                    linestyle = acq_linestyles.get(acq_type, "-")
                    marker = acq_markers.get(acq_type, "o")

                    line, = ax.plot(
                        iterations, mean_metric,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markevery=max(1, len(iterations) // n_markers),
                        markersize=5,
                        linewidth=1.5,
                        label=label,
                    )

                    if show_fill and n_seeds > 1:
                        ax.fill_between(
                            iterations,
                            mean_metric - uncertainty,
                            mean_metric + uncertainty,
                            color=color,
                            alpha=0.15,
                            edgecolor='none',
                            linewidth=0,
                        )

                    all_lines[label] = line

        # Plot baselines (e.g., RS)
        for baseline in baselines:
            if baseline == "RS":
                baseline_dir = results_root / f"d{dim}_RS"
                if not baseline_dir.exists():
                    print(f"Warning: {baseline_dir} not found, skipping")
                    continue

                result_files = list(baseline_dir.glob("*_seed*.pt"))
                if not result_files:
                    continue

                all_metric = []
                for rf in result_files:
                    try:
                        history = torch.load(rf, weights_only=False)
                        if metric in history.get("metrics", {}):
                            all_metric.append(history["metrics"][metric])
                    except Exception as e:
                        print(f"Warning: Failed to load {rf}: {e}")

                if all_metric:
                    all_metric = np.array(all_metric)
                    mean_metric = all_metric.mean(axis=0)
                    iterations = np.arange(len(mean_metric))
                    n_seeds = len(all_metric)
                    if use_stderr and n_seeds > 1:
                        uncertainty = all_metric.std(axis=0) / np.sqrt(n_seeds)
                    else:
                        uncertainty = all_metric.std(axis=0)

                    line, = ax.plot(
                        iterations, mean_metric,
                        color=rs_color,
                        linestyle='--',
                        linewidth=1.5,
                        marker='s',
                        markevery=max(1, len(iterations) // n_markers),
                        markersize=5,
                        label='RS',
                    )
                    if show_fill and n_seeds > 1:
                        ax.fill_between(
                            iterations,
                            mean_metric - uncertainty,
                            mean_metric + uncertainty,
                            color=rs_color,
                            alpha=0.15,
                            edgecolor='none',
                            linewidth=0,
                        )
                    all_lines['RS'] = line

        ax.set_xlabel("Iteration")
        if dim_idx == 0:
            metric_label = {
                "weighted_mse": "Weighted MSE",
                "mse": "MSE",
                "nlml": "NLML",
                "weighted_nlml": "Weighted NLML",
            }.get(metric, metric.upper())
            ax.set_ylabel(metric_label)
        ax.set_title(f"$d = {dim}$")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Row 2: Bar chart of SMC timing (acq_type grouped, particle count on x)
    bar_configs = [(kernel, acq_type) for kernel in kernels for acq_type in acq_types]
    n_bars = len(bar_configs)

    for dim_idx, dim in enumerate(dims):
        ax = axes[1, dim_idx]

        x_labels = []
        bar_data = {config: {"mean": [], "std": []} for config in bar_configs}

        for pmult_idx, pmult in enumerate(particle_mults):
            n_particles = pmult * dim
            x_labels.append(f"{pmult}·d")

            for kernel, acq_type in bar_configs:
                data = timing_data[dim][pmult][kernel][acq_type]
                bar_data[(kernel, acq_type)]["mean"].append(np.mean(data) if data else 0)
                bar_data[(kernel, acq_type)]["std"].append(np.std(data) if data else 0)

        x = np.arange(len(x_labels))
        total_width = 0.75
        width = total_width / max(n_bars, 1)

        for i, (kernel, acq_type) in enumerate(bar_configs):
            offset = (i - (n_bars - 1) / 2) * width
            acq_display = r"AB-$\bar{\mathcal{X}}$" if acq_type == "AB" else r"TS-$\bar{\mathcal{X}}$"
            # Bars for each particle count use the matching shade from the line plot
            bar_colors = [particle_color(acq_type, p_idx) for p_idx in range(len(particle_mults))]
            ax.bar(x + offset, bar_data[(kernel, acq_type)]["mean"], width,
                   yerr=bar_data[(kernel, acq_type)]["std"],
                   label=acq_display, color=bar_colors, capsize=2,
                   alpha=0.85, edgecolor='#333333', linewidth=0.6)

        ax.set_xlabel("Particles")
        if dim_idx == 0:
            ax.set_ylabel("MCMC time (s/iter)")
        ax.set_title(f"$d = {dim}$")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.3, axis='y')

    # Shared legend at bottom: compact, grouped by method
    # Build handles: one per (acq_type, pmult) + RS if present
    legend_handles = []
    for acq_type in acq_types:
        acq_display = r"AB-$\bar{\mathcal{X}}$" if acq_type == "AB" else r"TS-$\bar{\mathcal{X}}$"
        marker = acq_markers.get(acq_type, "o")
        for pmult_idx, pmult in enumerate(particle_mults):
            color = particle_color(acq_type, pmult_idx)
            pmult_label = str(pmult)
            legend_handles.append(
                Line2D([0], [0], color=color, marker=marker, markersize=6,
                       linewidth=1.5, linestyle='-',
                       label=f"{acq_display}, {pmult_label}·d")
            )
    if "RS" in baselines:
        legend_handles.append(
            Line2D([0], [0], color=rs_color, marker='s', markersize=6,
                   linewidth=1.5, linestyle='--', label='RS')
        )

    if legend_handles:
        ncol = len(particle_mults) * len(acq_types) + (1 if "RS" in baselines else 0)
        ncol = min(ncol, 10)
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.015),
            ncol=ncol,
            fontsize=15,
            frameon=True, fancybox=True, framealpha=0.95,
            edgecolor="#CCCCCC",
            handletextpad=0.5, columnspacing=1.3, handlelength=2.0,
            borderpad=0.7,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(h_pad=0.8, w_pad=1.3)
    if legend_handles:
        plt.subplots_adjust(bottom=0.16, hspace=0.4)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def spotlight_comparison(
    results_dir: Union[str, Path],
    methods: list,
    seed: int = 0,
    iteration: int = -1,
    n_grid: int = 100,
    figsize: tuple = None,
    save_path: Union[str, Path] = None,
    dpi: int = 150,
    cmap_error: str = "Reds",
    contour_levels: int = 25,
    suptitle: str = None,
    weighted: bool = True,
    density_fn: str = "exp",
    show_contour_labels: bool = True,
    plot_density: bool = False,
    log_error_scale: bool = False,
    title_fontsize: int = 12,
):
    """
    Create a spotlight comparison figure showing ground truth and weighted squared errors.

    Creates a 1-row plot with:
    - First panel: Ground truth function with contour lines
    - Remaining panels: Pointwise weighted squared error for each method with contour overlay
    - Single shared colorbar on the right for direct comparison

    Args:
        results_dir: Path to results directory (e.g., "exps/results/.../Gramacy_2d/matern52")
        methods: List of method names to compare (e.g., ["RS", "US", "IMSE", "AB-SID-iVAR"])
        seed: Random seed to use for loading results
        iteration: Which iteration to visualize (-1 for final)
        n_grid: Grid resolution for visualization
        figsize: Figure size (auto-computed if None)
        save_path: Path to save figure (optional)
        dpi: DPI for saved figure
        cmap_error: Colormap for prediction error
        contour_levels: Number of contour levels
        suptitle: Super title for the figure
        weighted: If True, weight squared errors by density (sqrt(density) * squared_error)
        density_fn: Density function type - "exp" for exp(f), "exp_neg" for exp(-f)
        show_contour_labels: If True, show values on contour lines
        plot_density: If True, plot density (exp(f) or exp(-f)) instead of raw f values
        log_error_scale: If True, use log scale for error colormap (better for large error ranges)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = spotlight_comparison(
        ...     "exps/results/formal_benchmark/continuous/Gramacy_2d/matern52",
        ...     methods=["RS", "US", "IMSE", "AB-SID-iVAR", "TS-SID-iVAR"],
        ...     seed=0,
        ...     weighted=True,
        ... )
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from botorch.models import SingleTaskGP

    results_dir = Path(results_dir)

    # Load first result to get problem info
    first_method = methods[0]
    first_file = _find_result_file(results_dir, first_method, seed)
    first_result = torch.load(first_file, weights_only=False)

    # Get problem info
    problem_name = first_result.get("problem_name", "Unknown")
    problem_kwargs = first_result.get("problem_kwargs", {})

    # Get test function
    test_fn = _get_test_function_for_viz(problem_name, problem_kwargs)
    bounds = test_fn.bounds
    dim = bounds.shape[1]

    if dim != 2:
        raise ValueError(f"spotlight_comparison only supports 2D problems, got dim={dim}")

    # Create visualization grid
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), n_grid, dtype=torch.float64)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), n_grid, dtype=torch.float64)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    # Evaluate ground truth
    with torch.no_grad():
        y_true = test_fn(X_grid).reshape(n_grid, n_grid)

    X1_np, X2_np = X1.numpy(), X2.numpy()
    y_true_np = y_true.numpy()

    # Transform to density if requested
    if plot_density:
        if density_fn == "exp":
            y_true_for_plot = np.exp(y_true_np)
        else:  # "exp_neg"
            y_true_for_plot = np.exp(-y_true_np)
    else:
        y_true_for_plot = y_true_np

    # Compute density weights for weighted error
    if weighted:
        if density_fn == "exp":
            # f is a log-density, so exp(f) is the density
            density = torch.exp(y_true)
        else:  # "exp_neg"
            # For functions like Branin: exp(-f) is the density
            density = torch.exp(-y_true)
        # Normalize density to [0, 1] for visualization
        density = density / density.max()
        density_np = density.numpy()
        # Use sqrt for weighting to preserve sign: sqrt(w) * error
        weight_np = np.sqrt(density_np)
    else:
        weight_np = np.ones_like(y_true_np)

    # Setup figure
    n_panels = 1 + len(methods)  # ground truth + methods
    if figsize is None:
        # Smaller panels since we're using a shared colorbar
        figsize = (3.2 * n_panels + 0.8, 3.5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Compute fixed contour levels from ground truth for consistent comparison
    y_min, y_max = y_true_for_plot.min(), y_true_for_plot.max()
    ground_truth_levels = np.linspace(y_min, y_max, contour_levels)

    # Panel 1: Ground truth (no colorbar to match other panels' size)
    ax = axes[0]
    # Compute density for filled background
    if density_fn == "exp":
        density_for_plot = np.exp(y_true_np)
    else:  # "exp_neg"
        density_for_plot = np.exp(-y_true_np)
    # Filled background showing density (white=0, colored=high density)
    ax.contourf(X1_np, X2_np, density_for_plot, levels=50, cmap='Blues', alpha=0.9)
    # Contour lines showing f values (for reference)
    contour = ax.contour(X1_np, X2_np, y_true_for_plot, levels=ground_truth_levels, colors='black', linewidths=0.5, alpha=0.7)
    if show_contour_labels:
        ax.clabel(contour, inline=True, fontsize=6, fmt='%.1f')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Ground Truth', fontsize=title_fontsize)
    ax.set_aspect('equal')

    # Remaining panels: Methods
    # Find global error range for consistent colorbar
    all_errors = []
    method_results = {}

    for method in methods:
        try:
            result_file = _find_result_file(results_dir, method, seed)
            result = torch.load(result_file, weights_only=False)
            method_results[method] = result
        except FileNotFoundError:
            print(f"Warning: Could not find results for {method} seed={seed}")
            method_results[method] = None
            continue

        # Get iteration - result["X"] is a list of tensors per iteration
        n_iters = len(result["X"])
        iter_idx = iteration if iteration >= 0 else n_iters - 1

        # Get training data up to this iteration
        # result["X"] is list of X tensors, result["final_X"] is the final combined tensor
        if iter_idx >= n_iters - 1:
            # Use final data
            train_X = result["final_X"]
            train_Y = result["final_Y"]
        else:
            # Stack X tensors up to iter_idx
            train_X = torch.cat(result["X"][:iter_idx + 1], dim=0)
            train_Y = torch.cat(result["Y"][:iter_idx + 1], dim=0)

        # Load saved hyperparameters and model config from result file
        saved_hyperparams = result["model_hyperparameters"][iter_idx]
        model_config = result.get("model_config", {})

        # Build model (don't fit - use saved hyperparameters)
        from sidal.benchmarking.model_builder import build_and_fit_gp
        gp = build_and_fit_gp(
            train_X=train_X,
            train_Y=train_Y,
            input_bounds=bounds,
            kernel=model_config["kernel"],
            use_input_transform=model_config["use_input_transform"],
            use_outcome_transform=model_config["use_outcome_transform"],
            use_default_kernel_priors=model_config.get("use_default_kernel_priors", False),
            fit_model=False,
        )

        # Set saved hyperparameters (handle ScaleKernel wrapper if present)
        if hasattr(gp.covar_module, "base_kernel"):
            gp.covar_module.base_kernel.lengthscale = saved_hyperparams["lengthscale"]
            if "outputscale" in saved_hyperparams:
                gp.covar_module.outputscale = saved_hyperparams["outputscale"]
        else:
            # No ScaleKernel (e.g., when use_default_kernel_priors=True with outcome_transform)
            gp.covar_module.lengthscale = saved_hyperparams["lengthscale"]

        gp.likelihood.noise_covar.noise = torch.tensor(saved_hyperparams["noise"])

        # Set mean constant if available
        if "mean_constant" in saved_hyperparams and hasattr(gp.mean_module, "constant"):
            gp.mean_module.constant.data.fill_(saved_hyperparams["mean_constant"])

        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X_grid)
            pred_mean = posterior.mean.squeeze().reshape(n_grid, n_grid)

        # Compute error (optionally weighted)
        raw_error = (pred_mean - y_true).numpy()
        squared_error = raw_error ** 2
        error = weight_np * squared_error  # sqrt(density) * squared_error for weighted MSE
        all_errors.append(error)
        method_results[method] = {
            "result": result,
            "train_X": train_X,
            "pred_mean": pred_mean,
            "error": error,
            "raw_error": raw_error,
            "iter_idx": iter_idx,
        }

    # Determine error range from all algorithms
    if all_errors:
        vmax = max(e.max() for e in all_errors)
        if log_error_scale:
            # For log scale, use 1st percentile of positive values to avoid numerical underflow
            all_positive = np.concatenate([e[e > 1e-10].flatten() for e in all_errors])
            if len(all_positive) > 0:
                vmin = max(np.percentile(all_positive, 1), vmax * 1e-6)
            else:
                vmin = vmax * 1e-6
        else:
            vmin = 0
    else:
        vmin, vmax = 1e-6 if log_error_scale else 0, 1

    # Set up normalization for colormap
    if log_error_scale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Mapping from abbreviations to full display names
    display_names = {
        "RS": "Random Sampling",
        "US": "Uncertainty Sampling",
        "IMSE": "Integrated MSE",
    }

    # Plot each method (store last contourf for shared colorbar)
    last_cf = None
    for i, method in enumerate(methods):
        ax = axes[i + 1]

        display_name = display_names.get(method, method)
        if method_results.get(method) is None or isinstance(method_results[method], type(None)):
            ax.text(0.5, 0.5, f'{display_name}\nNo data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(display_name, fontweight='bold', fontsize=title_fontsize)
            continue

        data = method_results[method]
        if isinstance(data, dict) and "error" in data:
            error = data["error"]
            train_X = data["train_X"]
            iter_idx = data["iter_idx"]

            # Prediction error (filled) - no individual colorbar
            if log_error_scale:
                # Use log-spaced levels for log scale
                error_clipped = np.clip(error, vmin, vmax)
                log_levels = np.logspace(np.log10(vmin), np.log10(vmax), 50)
                cf = ax.contourf(X1_np, X2_np, error_clipped, levels=log_levels, cmap=cmap_error, norm=norm)
            else:
                cf = ax.contourf(X1_np, X2_np, error, levels=50, cmap=cmap_error, vmin=vmin, vmax=vmax)
            last_cf = cf  # Store for shared colorbar

            # GP prediction contour overlay (lines only) - same levels as ground truth for direct comparison
            pred_mean_np = data["pred_mean"].numpy()
            # Transform to density if requested
            if plot_density:
                if density_fn == "exp":
                    pred_for_plot = np.exp(pred_mean_np)
                else:  # "exp_neg"
                    pred_for_plot = np.exp(-pred_mean_np)
            else:
                pred_for_plot = pred_mean_np
            contour_gp = ax.contour(X1_np, X2_np, pred_for_plot, levels=ground_truth_levels, colors='black', linewidths=0.5, alpha=0.7)
            if show_contour_labels:
                ax.clabel(contour_gp, inline=True, fontsize=6, fmt='%.1f')

            # Training points
            train_X_np = train_X.numpy()
            ax.scatter(train_X_np[:, 0], train_X_np[:, 1], c='black', s=15,
                       marker='o', edgecolors='white', linewidths=0.5, zorder=10, alpha=0.8)

            ax.set_xticks([])
            ax.set_yticks([])
            wmse = error.mean()  # Mean of weighted squared error
            ax.set_title(f'{display_name}\n(wmse={wmse:.4f})', fontsize=title_fontsize)
            ax.set_aspect('equal')

    # Only add suptitle if explicitly provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    # Add shared colorbar on the right for all method panels
    plt.tight_layout()
    if last_cf is not None:
        # Make room for colorbar on the right
        fig.subplots_adjust(right=0.90)
        # Add colorbar axes on the right side (wider for better visibility)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        error_label = 'Weighted Squared Error' if weighted else 'Squared Error'
        cbar = fig.colorbar(last_cf, cax=cbar_ax, label=error_label)
        # Only set clim if not using log scale (norm handles it otherwise)
        if norm is None:
            cbar.mappable.set_clim(vmin, vmax)
        else:
            # Use nice round ticks for log scale
            from matplotlib.ticker import LogLocator, LogFormatterMathtext
            cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.label.set_size(10)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def _get_test_function_for_viz(problem_name: str, problem_kwargs: dict = None):
    """Get test function for visualization, supporting custom 1D functions."""
    from sidal.test_functions import benchmark_test_functions as test_functions
    from sidal.test_functions import Forrester, Gramacy1D, Higdon

    if problem_kwargs is None:
        problem_kwargs = {}

    custom_functions = {
        "Forrester": Forrester,
        "Gramacy1D": Gramacy1D,
        "Higdon": Higdon,
    }

    # Check custom functions first
    if problem_name in custom_functions:
        func_class = custom_functions[problem_name]
        kwargs_clean = {k: v for k, v in problem_kwargs.items() if k != 'seed'}
        try:
            return func_class(**kwargs_clean)
        except TypeError:
            kwargs_clean.pop('dim', None)
            return func_class(**kwargs_clean)

    # Fall back to botorch test functions
    if hasattr(test_functions, problem_name):
        func_class = getattr(test_functions, problem_name)
        kwargs_clean = {k: v for k, v in problem_kwargs.items() if k not in ['seed']}
        try:
            return func_class(**kwargs_clean)
        except TypeError:
            kwargs_clean.pop('dim', None)
            return func_class(**kwargs_clean)

    raise ValueError(f"Unknown problem: {problem_name}")
