"""
ThesisMSc_FINAL.ipynb – Headless OLS Replication Runner
Runs the full notebook pipeline with use_quantreg=False (strict A&B 2016)
and saves all plots as PNG files to results_ols/plots/
"""
import matplotlib
matplotlib.use('Agg')  # Headless backend – no GUI windows

import pandas as pd
import numpy as np
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covar_engine import (
    run_full_pipeline, print_summary_report,
    plot_dynamic_covar, plot_ranking_shift, plot_feature_importance, plot_forward_covar_fit,
    run_sensitivity_analysis, plot_sensitivity_heatmap,
    make_unconditional_ranking_table, make_conditional_ranking_table,
    make_forward_covar_table, make_backtest_table,
    TICKERS
)

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH = os.path.expanduser('~/Downloads/thesis_full_df_backup_final.csv')
OUT_DIR = "results_ols"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print(f"Loading dataframe from: {CSV_PATH}")
full_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

# =============================================================================
# PHASE 1: FULL PIPELINE (OLS, Expanding Window)
# =============================================================================
print("\n" + "="*70)
print("  PHASE 1: FULL FORWARD-CoVaR PIPELINE (OLS REPLICATION)")
print("="*70)

results = run_full_pipeline(
    full_df,
    q=0.05,
    horizon=1,
    scale_features=True,
    use_expanding=True,
    use_quantreg=False,   # <-- STRICT A&B (2016) OLS
    verbose=True
)

print_summary_report(results)

# =============================================================================
# PHASE 2: SAVE ALL PLOTS
# =============================================================================
print("\n" + "="*70)
print("  PHASE 2: GENERATING AND SAVING ALL PLOTS")
print("="*70)

# Figure 1: Time-Varying ΔCoVaR
print("  Saving: plot_dynamic_covar...")
plot_dynamic_covar(results, save_path=os.path.join(PLOT_DIR, "fig1_dynamic_covar_ols.png"))

# Figure 2: Forward-CoVaR Fit per coin
for t in TICKERS:
    if "forward" in results.get(t, {}):
        print(f"  Saving: plot_forward_covar_fit ({t.upper()})...")
        plot_forward_covar_fit(results, t, save_path=os.path.join(PLOT_DIR, f"fig2_forward_fit_{t}_ols.png"))

# Figure 3: Ranking Shift (Slope Chart)
print("  Saving: plot_ranking_shift...")
plot_ranking_shift(results, save_path=os.path.join(PLOT_DIR, "fig3_ranking_shift_ols.png"))

# Figure 4: Feature Importance
print("  Saving: plot_feature_importance...")
plot_feature_importance(results, save_path=os.path.join(PLOT_DIR, "fig4_feature_importance_ols.png"))

# =============================================================================
# PHASE 3: SENSITIVITY ANALYSIS (Expanding, OLS)
# =============================================================================
print("\n" + "="*70)
print("  PHASE 3: SENSITIVITY ANALYSIS (EXPANDING + OLS)")
print("="*70)

sens_df = run_sensitivity_analysis(
    full_df,
    quantiles=[0.01, 0.05, 0.10],
    horizons=[1, 5, 10],
    scale_features=True,
    use_expanding=True,
    use_quantreg=False   # <-- STRICT A&B (2016) OLS
)

print("\n--- Sensitivity Analysis Results ---")
print(sens_df.to_string())

# Save sensitivity heatmap
import matplotlib.pyplot as plt
print("  Saving: sensitivity_heatmap...")
import seaborn as sns

coins = sens_df.index.get_level_values('Coin').unique()
fig, axes = plt.subplots(1, len(coins), figsize=(5 * len(coins), 4), sharey=True)
if len(coins) == 1:
    axes = [axes]
for ax, coin in zip(axes, coins):
    subset = sens_df.xs(coin, level='Coin')
    pivot = subset.reset_index().pivot_table(index='Quantile', columns='Horizon', values='Loss_OOS')
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".4f", ax=ax, cbar_kws={'label': 'MSE OOS'})
    ax.set_title(f"{coin} - MSE OOS")
    ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "fig5_sensitivity_heatmap_ols.png"), dpi=150)
plt.close()

# =============================================================================
# PHASE 4: EXPORT ALL TABLES TO CSV
# =============================================================================
print("\n" + "="*70)
print("  PHASE 4: EXPORTING ALL RESULTS TO CSV")
print("="*70)

make_unconditional_ranking_table(results).to_csv(f"{OUT_DIR}/table1_unconditional_ranking.csv")
make_conditional_ranking_table(results).to_csv(f"{OUT_DIR}/table2_conditional_ranking.csv")
make_forward_covar_table(results).to_csv(f"{OUT_DIR}/table3_forward_covar.csv")
make_backtest_table(results).to_csv(f"{OUT_DIR}/table4_backtests.csv")
sens_df.to_csv(f"{OUT_DIR}/sensitivity_analysis.csv")

print(f"Success! All tables saved to '{OUT_DIR}/'")
print(f"All plots saved to '{PLOT_DIR}/'")

print("\n" + "="*70)
print("  OLS REPLICATION COMPLETE")
print("="*70)
