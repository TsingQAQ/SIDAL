"""Figure 4 (Section 5.2): weighted MSE on the four PES fitting tasks.

Reads exps/results/formal_benchmark/continuous/ and writes
figures/applied_benchmark_30reps.pdf, with per-panel iteration truncation.

Uses the canonical generate_convergence_curve_for_paper to preserve paper
styling, then post-processes per-axis xlim.
"""
from pathlib import Path
from sidal.benchmarking.visualization import (
    generate_convergence_curve_for_paper,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "exps/results/formal_benchmark/continuous"
OUT = ROOT / "figures/applied_benchmark_30reps.pdf"

panels = [
    (r"H$_2$ on Cu ($d=2$)",           "H2Cu_2d/matern52",            100),
    (r"H on Cu Cluster ($d=3$)",       "HOnCu13_3d/matern52",         150),
    (r"Si Crystal ($d=3$)",            "SiSW_3d/matern52",            150),
    (r"H$_2$O on Pt ($d=6$)",          "H2OonPt111_Slab_6d/matern52", 300),
]
exp_names = [p[1] for p in panels]
exp_titles = [p[0] for p in panels]
xlims = [p[2] for p in panels]
methods = ["RS", "US", "EPIG", "IMSE", "HAL", "TS-SID-iVAR_EWV", "AB-SID-iVAR_EWV"]

fig = generate_convergence_curve_for_paper(
    exp_root=str(EXP_ROOT),
    exp_names=exp_names,
    methods=methods,
    metric="weighted_mse",
    max_cols=4,
    figsize_per_panel=(6, 4.2),
    legend_ncol=7,
    linewidth=1.8,
    markersize=6,
    exp_titles=exp_titles,
    save_path=None,
)

# Post-process: per-panel x-axis truncation
panel_axes = [ax for ax in fig.axes if ax.has_data()]
for ax, xmax in zip(panel_axes, xlims):
    ax.set_xlim(0, xmax)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}")
