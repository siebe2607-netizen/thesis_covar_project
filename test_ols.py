import pandas as pd
from covar_engine import build_features, estimate_rolling_delta_covar, estimate_forward_covar, estimate_forward_covar_expanding
import os

CSV_PATH = os.path.expanduser('~/Downloads/thesis_full_df_backup_final.csv')
print("Loading subset of data for OLS syntax test...")
# Load a tiny 800-row slice of the data so it runs in 2 seconds
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True).iloc[-800:]
df = build_features(df)

print("Calculating rolling Delta-CoVaR...")
rolling = estimate_rolling_delta_covar(df, 'btc', q=0.05, window=100)

print("Running pure OLS block (Expanding)...")
fwd = estimate_forward_covar_expanding(df, 'btc', rolling, min_train_size=200, use_quantreg=False)

print(f"\nSUCCESS! Engine correctly flipped to: {fwd['loss_name']}")
print(f"R-Squared = {fwd['r2_is']:.4f}")
print(f"MSE IS = {fwd['loss_is']:.6f}")
print(f"MSE OOS = {fwd['loss_oos']:.6f}")
