import sys
import os
import pandas as pd
import numpy as np

# Attach the core mathematical engine safely
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from covar_engine import estimate_conditional_covar, get_market_state, TICKERS, QUANTILE

CSV_PATH = os.path.expanduser('~/Downloads/thesis_full_df_backup_final.csv')
if not os.path.exists(CSV_PATH):
    print(f"Dataset not found at {CSV_PATH}. Make sure to export it first.")
    sys.exit(1)

print("Loading dataset for Regime-Switching analysis...")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

# 1. Establish the Shifted Variables First
# We must shift market states on the continuous timeline BEFORE splitting into regimes
# so that the day t prediction correctly aligns with the state of day t-1.
# (The conditional engine does this internally, but to define our regimes safely, we do it here).

M_stat = get_market_state(df)
df['M_vol_shifted'] = M_stat['M_vol'].shift(1)

# 2. Define the Regimes
# We can computationally split "Bear / High Distress" vs "Bull / Normal"
# using the median of historical macro volatility.
median_vol = df['M_vol_shifted'].median()

# Split the master dataframe
df_high_vol = df[df['M_vol_shifted'] >= median_vol]
df_low_vol  = df[df['M_vol_shifted'] < median_vol]

print(f"\nRegime Split:")
print(f"  Low Volatility (Bull) days: {len(df_low_vol)}")
print(f"  High Volatility (Bear) days: {len(df_high_vol)}")

# 3. Evaluate conditional coefficients
# We want to see how the coefficient for gamma (Coin Return) changes!
# Does a 1% drop in BTC cause a larger systemic drop in a Bear market than a Bull market?

results = []

print("\n" + "="*70)
print("  REGIME-SWITCHING CONDITIONAL CoVaR ANALYSIS")
print("="*70)

for t in TICKERS:
    try:
        # Run the conditional regression on the Low Volatility timeframe
        cond_low = estimate_conditional_covar(df_low_vol, t, q=QUANTILE)
        gamma_low_vol = cond_low.attrs["covar_params"][f"{t}_ret"]
        mean_dcovar_low = cond_low["DeltaCoVaR_t"].mean()
        
        # Run the conditional regression on the High Volatility timeframe
        cond_high = estimate_conditional_covar(df_high_vol, t, q=QUANTILE)
        gamma_high_vol = cond_high.attrs["covar_params"][f"{t}_ret"]
        mean_dcovar_high = cond_high["DeltaCoVaR_t"].mean()
        
        results.append({
            "Coin": t.upper(),
            "Bull_ΔCoVaR": mean_dcovar_low,
            "Bear_ΔCoVaR": mean_dcovar_high,
            "Gamma_Bull": gamma_low_vol,
            "Gamma_Bear": gamma_high_vol
        })
        
        print(f"\n[{t.upper()}] Systemic Importance (Gamma Coefficient):")
        print(f"  Normal Market : {gamma_low_vol:.4f}")
        print(f"  Distressed    : {gamma_high_vol:.4f}")
        
    except Exception as e:
        print(f"Error processing {t}: {e}")

# 4. Export Table
final_df = pd.DataFrame(results).set_index("Coin")
final_df['Risk_Amplifier'] = (final_df['Bear_ΔCoVaR'] / final_df['Bull_ΔCoVaR']) - 1

results_path = os.path.join(os.path.dirname(__file__), "results", "regime_switching_analysis.csv")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
final_df.to_csv(results_path)

print("\n--- Summary Data Generated ---")
print(final_df[['Bull_ΔCoVaR', 'Bear_ΔCoVaR', 'Risk_Amplifier']].round(4))
print(f"\n=> Full statistics successfully exported to: {results_path}")
