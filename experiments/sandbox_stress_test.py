import sys
import os
import pandas as pd

# Add the parent directory to the path so we can import the original engine logic
# without risking modifying it.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from covar_engine import run_full_pipeline, make_unconditional_ranking_table

# 1. Load the pristine data
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../thesis_full_df_backup_final.csv')
if not os.path.exists(CSV_PATH):
    print(f"Dataset not found at {CSV_PATH}. Please ensure it is exported.")
    sys.exit(1)

print("Loading pristine dataset...")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

# 2. Establish the Baseline Scenario (Normal Conditions)
# You could run this to have a control group. Here we will just calculate
# the baseline unconditional ranking.
print("\n[BASELINE] Analysing normal macroeconomic conditions...")
baseline_results = run_full_pipeline(df, scale_features=True, use_expanding=False, verbose=False)
baseline_table = make_unconditional_ranking_table(baseline_results)
print(baseline_table)


# 3. Set up the Stress Scenario
# Let's see what happens to the rankings if Market Volatility artificially explodes.
print("\n" + "="*60)
print("  STRESS TEST: APPLYING 400% SHOCK TO MACRO VOLATILITY")
print("="*60)

stress_df = df.copy()

# Ensure the column actually exists in the data, then shock it:
if 'M_vol' in stress_df.columns:
    stress_df['M_vol'] = stress_df['M_vol'] * 4.0
else:
    print("Warning: 'M_vol' column not found. The macro volatility shock was not applied.")

# 4. Run the engine on the stressed data
stress_results = run_full_pipeline(stress_df, scale_features=True, use_expanding=False, verbose=False)
stress_table = make_unconditional_ranking_table(stress_results)

print("\n[STRESSED] Analysing high volatility conditions...")
print(stress_table)

# 5. Save experimental output
results_path = os.path.join(os.path.dirname(__file__), "results", "stress_test_comparison.csv")

# Join baseline and stress table for a clean comparison
comparison = baseline_table[['ΔCoVaR']].rename(columns={'ΔCoVaR': 'Baseline_ΔCoVaR'})
comparison = comparison.join(stress_table[['ΔCoVaR']].rename(columns={'ΔCoVaR': 'Stressed_ΔCoVaR'}))
comparison['Shock_Impact'] = comparison['Stressed_ΔCoVaR'] - comparison['Baseline_ΔCoVaR']

comparison.to_csv(results_path)
print(f"\n=> Exported final comparison to {results_path}")
