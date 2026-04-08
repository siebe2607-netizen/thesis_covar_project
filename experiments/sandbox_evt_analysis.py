import sys
import os
import pandas as pd
import numpy as np
import scipy.stats as stats

# Attach the core analytical framework
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from covar_engine import TICKERS

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../thesis_full_df_backup_final.csv')
if not os.path.exists(CSV_PATH):
    print(f"Dataset not found at {CSV_PATH}.")
    sys.exit(1)

print("Loading dataset for Extreme Value Theory (EVT) analysis...")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

print("\n" + "="*70)
print("  EXTREME VALUE THEORY (EVT) SYSTEMIC RISK STRESS TEST")
print("="*70)
print("Methodology: Peaks-Over-Threshold (Generalized Pareto Distribution)")
print("Modeling the absolute 0.1% extreme tail of the cryptocurrency market.\n")

evt_results = []

for t in TICKERS:
    try:
        coin_ret = f"{t}_ret"
        sys_ret = f"sys_ret_excl_{t}"
        
        # We need both columns to exist
        if coin_ret not in df.columns or sys_ret not in df.columns:
            continue
            
        # 1. Isolate the Distress events for the specific coin
        # Let's say "Distress" is when the coin's return is in its worst 5% historical days.
        coin_distress_threshold = df[coin_ret].quantile(0.05)
        df_distress = df[df[coin_ret] <= coin_distress_threshold].copy()
        
        # 2. Extract the System Losses during these distressed periods
        # We multiply by -1 because EVT naturally models "maximums" (positive losses)
        system_losses = -df_distress[sys_ret].dropna()
        
        # 3. Establish the "Threshold" (u) for the Peaks-Over-Threshold method
        # We will look at the worst 20% of system drops *during* coin distress
        u = system_losses.quantile(0.80)
        
        # Extract Exceedances (how far past the threshold did the system crash?)
        exceedances = system_losses[system_losses > u] - u
        
        # 4. Fit the Generalized Pareto Distribution (GPD)
        # c is the shape parameter (tail index), scale is the volatility of the tail
        c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        
        # 5. Calculate Extreme 0.1% Systemic VaR using the EVT formula
        # P(X > x) = (N_u / n) * (1 + c * (x - u) / scale)^(-1/c)
        # We solve for x at p = 0.001 (0.1% probability)
        
        n = len(system_losses)
        n_u = len(exceedances)
        p_extreme = 0.001  # 0.1% Tail
        
        # EVT VaR formula derivation
        if c != 0:
            evt_var = u + (scale / c) * (((n / n_u) * p_extreme)**(-c) - 1)
        else:
            evt_var = u - scale * np.log((n / n_u) * p_extreme)
            
        # Reconvert back to negative return perspective
        evt_systemic_crash = -evt_var
        
        evt_results.append({
            "Coin": t.upper(),
            "Empirical_5%_Crash": -system_losses.quantile(0.50), # Median loss during distress
            "EVT_0.1%_Doomsday_Crash": evt_systemic_crash,
            "GPD_Shape_(Tail_Fatness)": c
        })
        
        print(f"[{t.upper()}] EVT Computed.")
        print(f"  -> Threshold (u)       : {u:.4f} loss")
        print(f"  -> Tail Index (Shape)  : {c:.4f} (positive means fat-tailed!)")
        print(f"  -> 0.1% Doomsday Crash : {evt_systemic_crash*100:.2f}% System Drop")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error modeling EVT for {t}: {e}")

# Compile and export
final_evt = pd.DataFrame(evt_results).set_index("Coin")
final_evt = final_evt.sort_values("EVT_0.1%_Doomsday_Crash")  # Sort by worst doomsday scenario

results_path = os.path.join(os.path.dirname(__file__), "results", "evt_tail_risk_analysis.csv")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
final_evt.to_csv(results_path)

print("\n=> Exported EVT extreme tail modeling metrics to:")
print(f"   {results_path}")
