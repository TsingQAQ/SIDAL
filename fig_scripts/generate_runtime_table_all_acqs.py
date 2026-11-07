#!/usr/bin/env python
"""
Generate runtime table with ALL acquisition functions (no breakdown).

Same layout as generate_runtime_profile.py but with every acq as a flat column.
Rows are grouped by dimension (aggregated across problems of the same d),
split into Continuous and Discrete sections.

Usage:
  python fig_scripts/generate_runtime_table_all_acqs.py
"""

import argparse
import glob
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Display name -> file pattern
METHODS = {
    "RS": "RS",
    "US": "US",
    "IMSE": "IMSE",
    "EPIG": "EPIG",
    "GHAL": "HAL",
    r"AB-$\bar{\mathcal{X}}$": "AB-SID-iVAR_EWV_zeroth_order_analytic",
    "AB-NoC": "AB-SID-iVAR_NoC_zeroth_order_analytic",
    r"TS-$\bar{\mathcal{X}}$": "TS-SID-iVAR_EWV_thompson_sampling",
    "TS-NoC": "TS-SID-iVAR_NoC_thompson_sampling",
    r"$\mu$-$\bar{\mathcal{X}}$": "mu-SID-iVAR_EWV",
    r"$\mu$-NoC": "mu-SID-iVAR_NoC",
}

# Alternative file patterns (archived results carry a legacy "_beta0.0" tag)
ALT_PATTERNS = {
    pattern: [f"{pattern}_beta0.0"] for pattern in METHODS.values()
}
ALT_PATTERNS["AB-SID-iVAR_EWV_zeroth_order_analytic"].append("AB-SID-iVAR-EWV_EWV")
ALT_PATTERNS["TS-SID-iVAR_EWV_thompson_sampling"].append("TS-SID-iVAR-EWV_EWV")

# Representative problems (one per dimension, matching generate_runtime_profile.py)
PROBLEMS_BY_DIM = {
    1: ["Gramacy_1d"],
    2: ["Branin_2d"],
    3: ["Hartmann_3d"],
    6: ["Hartmann_6d"],
}

DIMS = [1, 2, 3, 6]


def load_seed_means(
    results_dir: Path, problem: str, pattern: str,
    kernel: str = "matern52", mode: str = "continuous",
) -> Optional[np.ndarray]:
    """Return array of per-seed mean iteration times."""
    files = glob.glob(str(results_dir / mode / problem / kernel / f"{pattern}_seed*.pt"))
    if not files:
        for alt in ALT_PATTERNS.get(pattern, []):
            files = glob.glob(str(results_dir / mode / problem / kernel / f"{alt}_seed*.pt"))
            if files:
                break
    if not files:
        return None

    means = []
    for f in files:
        try:
            data = torch.load(f, weights_only=False, map_location="cpu")
            t = data.get("timing", {}).get("iteration_total", [])
            if t:
                means.append(np.mean(t))
        except Exception:
            continue
    return np.array(means) if means else None


def fmt(vals: Optional[np.ndarray]) -> str:
    if vals is None or len(vals) == 0:
        return "--"
    m, s = np.mean(vals), np.std(vals)
    if m < 0.01:
        return f"{m:.3f}$\\pm${s:.3f}"
    elif m < 1:
        return f"{m:.2f}$\\pm${s:.2f}"
    elif m < 100:
        return f"{m:.1f}$\\pm${s:.1f}"
    else:
        return f"{m:.0f}$\\pm${s:.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("exps/results/formal_benchmark"))
    parser.add_argument("--output", type=Path, default=Path("figures/runtime_all_acqs"))
    args = parser.parse_args()

    method_names = list(METHODS.keys())
    method_patterns = list(METHODS.values())

    # Collect data: aggregate across problems of the same dimension
    rows = []
    for mode in ["continuous", "discrete"]:
        for dim in DIMS:
            problems = PROBLEMS_BY_DIM[dim]
            row = {"mode": mode, "dim": dim}
            any_data = False

            for name, pattern in METHODS.items():
                # Pool all seed means across problems of this dimension
                all_means = []
                for prob in problems:
                    vals = load_seed_means(args.results_dir, prob, pattern, mode=mode)
                    if vals is not None:
                        all_means.extend(vals.tolist())
                row[name] = np.array(all_means) if all_means else None
                if all_means:
                    any_data = True

            if any_data:
                rows.append(row)

    # LaTeX
    n = len(method_names)
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Mean runtime per iteration (seconds, mean $\\pm$ std across seeds).}")
    lines.append("\\label{tab:runtime_all_acqs}")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lc" + "c" * n + "@{}}")
    lines.append("\\toprule")
    lines.append(" & ".join(["", "$d$"] + method_names) + " \\\\")
    lines.append("\\midrule")

    cur_mode = None
    for row in rows:
        if row["mode"] != cur_mode:
            if cur_mode is not None:
                lines.append("\\midrule")
            cur_mode = row["mode"]
            mode_label = "Continuous" if cur_mode == "continuous" else "Discrete"
        else:
            mode_label = ""

        vals = [mode_label, str(row["dim"])]
        for name in method_names:
            vals.append(fmt(row.get(name)))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output.with_suffix(".tex"), "w") as f:
        f.write(latex)
    print(f"Saved to {args.output.with_suffix('.tex')}")

    # CSV
    csv_rows = []
    for row in rows:
        cr = {"mode": row["mode"], "dim": row["dim"]}
        for name in method_names:
            v = row.get(name)
            cr[name] = f"{np.mean(v):.3f}" if v is not None else ""
        csv_rows.append(cr)
    pd.DataFrame(csv_rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Saved to {args.output.with_suffix('.csv')}")

    # PNG via matplotlib table
    import matplotlib.pyplot as plt

    # Build cell text and headers
    col_headers = ["", "$d$"] + method_names
    cell_text = []
    cur_mode = None
    for row in rows:
        if row["mode"] != cur_mode:
            cur_mode = row["mode"]
            mode_label = "Continuous" if cur_mode == "continuous" else "Discrete"
        else:
            mode_label = ""
        r = [mode_label, str(row["dim"])]
        for name in method_names:
            r.append(fmt(row.get(name)).replace("$\\pm$", "\u00b1"))
        cell_text.append(r)

    fig, ax = plt.subplots(figsize=(max(18, 1.5 * len(col_headers)), 0.5 * len(cell_text) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=[h.replace("$\\bar{\\mathcal{X}}$", "X\u0304").replace("$\\mu$", "\u03bc").replace("$d$", "d") for h in col_headers],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_headers)):
        table[0, j].set_facecolor("#E8E8E8")
        table[0, j].set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output.with_suffix('.png')}")
    plt.close(fig)

    print("\n" + latex)


if __name__ == "__main__":
    main()
