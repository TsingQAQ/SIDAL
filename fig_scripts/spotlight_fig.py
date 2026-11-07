from sidal.benchmarking.visualization import spotlight_comparison

fig = spotlight_comparison(
    # results_dir="exps/results/formal_benchmark/continuous/Branin_2d/matern52",
    results_dir="exps/results/formal_benchmark/continuous/Gramacy_2d/matern52",
    methods=["RS", "US", "IMSE", "AB-SID-iVAR"],
    seed=0,
    contour_levels=20, 
    show_contour_labels=False,  
    plot_density=False,
    log_error_scale=True,
    save_path="figures/spotlight_Gramacy_2d.pdf",
    weighted=True,  # Default - weight errors by density
    density_fn="exp_neg",  # density = exp(-f)
    iteration=40,
    title_fontsize=15,
)