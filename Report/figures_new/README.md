# Corrected Figures for Academic Manuscript

This directory contains corrected versions of figures from the original report,
addressing Hans's PDF comments and meeting feedback.

## Phase 1: Corrected Original Figures

These figures directly correspond to figures in `main (3).tex`, with Hans's requested corrections applied.

| New Figure | Original | Hans Comments Addressed |
|------------|----------|------------------------|
| `4_2k_mean_convergence_envelopes_corrected.png` | `4_2k_mean_convergence_envelopes.png` | Figure 7: Corrected values (Resample: 1.5%, FixedMean: 25%, Baseline: 89%) |
| `4_3a_resample_rmse_distributions_logscale.png` | `4_3a_resample_rmse_distributions.png` | #105: Log scale, model labels, noise levels |
| `4_4a_post_assimilation_rmse_logscale.png` | `4_4a_post_assimilation_rmse.png` | Log scale, h(x) notation |
| `4_5a_trajectory_fidelity_grid.png` | `4_5a_trajectory_fidelity_comparison.png` | **NEW**: Comprehensive grid (3 modes × 2 regimes × 4 noise levels = 24 panels), f_θ notation, improved line visibility |
| `4_5a_architecture_comparison_grid.png` | N/A | **NEW**: Architecture comparison (3 archs × 3 modes × 2 noise levels = 18 panels) in Resample regime |
| `4_5b_error_evolution_profiles_corrected.png` | `4_5b_error_evolution_profiles.png` | Regime stability labels |
| `4_6a_background_sampling_stability_corrected.png` | `4_6a_background_sampling_stability.png` | Stability metrics, dropout rates |
| `training_validation_loss_grid.png` | N/A | **NEW**: Training/validation loss grid (3 modes × 4 noise levels) |

## Phase 2: Insight Analysis Figures

Additional figures for deeper analysis:

| Figure | Purpose |
|--------|---------|
| `fig_resample_rmse_by_mode.png` | RMSE by observation mode comparison |
| `fig_regime_stability_comparison.png` | 94% FixedMean failure rate visualization |
| `fig_noise_sensitivity.png` | RMSE vs noise level trends |
| `fig_boxplot_log_scale.png` | Log-scale boxplots per Hans #105 |
| `fig_rmse_vs_rmdse.png` | RMSE vs RMdSE robust metric comparison |

## Academic Best Practices Applied

1. **Log scale for RMSE plots** (Hans comment #105)
2. **Grouped bar charts** instead of boxplots for single-value data
3. **Clear mathematical notation**: h(x), f_θ, σ_obs
4. **Consistent color schemes** per architecture/mode
5. **Value labels** on bars for precise reading
6. **Grid lines** for visual guidance
7. **Publication-quality DPI** (150)

## How to Regenerate

```bash
# All figures
python3 Report/scripts_generated/generate_all_corrected_figures.py

# Individual figures
python3 Report/scripts_generated/plot_4_3a_rmse_distributions_logscale.py
python3 Report/scripts_generated/plot_4_4a_obs_mode_rmse_logscale.py
python3 Report/scripts_generated/plot_4_5a_trajectory_grid.py  # 24-panel grid (modes × regimes × noise)
python3 Report/scripts_generated/plot_4_5a_architecture_comparison_grid.py  # 18-panel grid (archs × modes × noise)
python3 Report/scripts_generated/plot_4_5b_error_evolution.py
python3 Report/scripts_generated/plot_4_6a_stability.py
python3 Report/scripts_generated/plot_convergence_envelopes.py
python3 Report/scripts_generated/plot_loss_grid_train_val.py
```

## Key Findings from Figures

1. **FixedMean instability**: 94% of configurations produce divergent trajectories
2. **Observation mode ordering**: x² (non-linear) is hardest, xy provides most information
3. **Architecture independence**: MLP, GRU, LSTM show similar RMSE (inconclusive)
4. **Noise sensitivity**: Higher σ_obs generally increases RMSE but effect is modest
5. **Improvement is ~0%**: Network provides minimal improvement over background
