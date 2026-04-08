"""
Descriptive Statistics Generator
MSc Thesis: Measuring Systemic Risk in the Cryptocurrency Market

Generates publication-ready descriptive statistics tables for the full dataset.
Outputs are saved to results/descriptive_statistics/ as CSV files.
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_full_df_backup_final.csv')
OUT_DIR = "results/descriptive_statistics"
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["btc", "eth", "xrp", "bnb", "sol"]
TICKER_NAMES = {"btc": "Bitcoin", "eth": "Ethereum", "xrp": "XRP", "bnb": "BNB", "sol": "Solana"}

# ── Load Data ─────────────────────────────────────────────────────────────────
print(f"Loading dataframe from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"Total observations: {len(df)}")
print()

# =============================================================================
# TABLE A: Price & Return Statistics (per coin)
# =============================================================================
print("=" * 70)
print("  TABLE A: PRICE & RETURN SUMMARY STATISTICS")
print("=" * 70)

price_cols = {t: f"{t}_PriceUSD" for t in TICKERS}
ret_cols   = {t: f"{t}_ret" for t in TICKERS}

rows = []
for t in TICKERS:
    p = df[price_cols[t]].dropna()
    r = df[ret_cols[t]].dropna()

    row = {
        "Asset": TICKER_NAMES[t],
        "N": len(r),
        "Mean Price ($)": round(p.mean(), 2),
        "Min Price ($)": round(p.min(), 2),
        "Max Price ($)": round(p.max(), 2),
        "Mean Return (%)": round(r.mean() * 100, 4),
        "Std Dev (%)": round(r.std() * 100, 4),
        "Skewness": round(r.skew(), 4),
        "Kurtosis": round(r.kurt(), 4),
        "Min Return (%)": round(r.min() * 100, 4),
        "Max Return (%)": round(r.max() * 100, 4),
        "JB Statistic": round(stats.jarque_bera(r)[0], 2),
        "JB p-value": f"{stats.jarque_bera(r)[1]:.4e}",
    }
    rows.append(row)

table_a = pd.DataFrame(rows).set_index("Asset")
print(table_a.to_string())
table_a.to_csv(os.path.join(OUT_DIR, "tableA_price_return_stats.csv"))
print(f"  → Saved to {OUT_DIR}/tableA_price_return_stats.csv\n")

# =============================================================================
# TABLE B: Market & Macro State Variable Statistics
# =============================================================================
print("=" * 70)
print("  TABLE B: MARKET & MACRO STATE VARIABLE STATISTICS")
print("=" * 70)

state_vars = {
    "M_vol": "Market Volatility",
    "M_trend": "Market Trend",
    "M_fund": "Market Fundamentals",
    "M_fiat": "Stablecoin Supply (Fiat Proxy)",
    "M_volu": "Market Volume",
    "mkt_ret": "Market Return",
}

rows_b = []
for col, label in state_vars.items():
    if col in df.columns:
        s = df[col].dropna()
        rows_b.append({
            "Variable": label,
            "N": len(s),
            "Mean": round(s.mean(), 6),
            "Std Dev": round(s.std(), 6),
            "Min": round(s.min(), 6),
            "Max": round(s.max(), 6),
            "Skewness": round(s.skew(), 4),
            "Kurtosis": round(s.kurt(), 4),
        })

table_b = pd.DataFrame(rows_b).set_index("Variable")
print(table_b.to_string())
table_b.to_csv(os.path.join(OUT_DIR, "tableB_state_variables.csv"))
print(f"  → Saved to {OUT_DIR}/tableB_state_variables.csv\n")

# =============================================================================
# TABLE C: On-Chain Factor Statistics (per coin)
# =============================================================================
print("=" * 70)
print("  TABLE C: ON-CHAIN FACTOR STATISTICS")
print("=" * 70)

factor_suffixes = {
    "Ret": "Return Factor",
    "NetAct": "Network Activity",
    "Val": "Valuation (MVRV)",
    "Lev": "Leverage (OI Ratio)",
    "Liq": "Liquidity (Exchange Flow)",
    "fac_security": "Security (Hash Rate)",
}

rows_c = []
for t in TICKERS:
    for suffix, label in factor_suffixes.items():
        col = f"{t}_{suffix}"
        if col in df.columns:
            s = df[col].dropna()
            rows_c.append({
                "Asset": TICKER_NAMES[t],
                "Factor": label,
                "N": len(s),
                "Mean": round(s.mean(), 6),
                "Std Dev": round(s.std(), 6),
                "Min": round(s.min(), 6),
                "Max": round(s.max(), 6),
                "Skewness": round(s.skew(), 4),
                "Kurtosis": round(s.kurt(), 4),
            })

table_c = pd.DataFrame(rows_c).set_index(["Asset", "Factor"])
print(table_c.to_string())
table_c.to_csv(os.path.join(OUT_DIR, "tableC_onchain_factors.csv"))
print(f"  → Saved to {OUT_DIR}/tableC_onchain_factors.csv\n")

# =============================================================================
# TABLE D: Return Correlation Matrix
# =============================================================================
print("=" * 70)
print("  TABLE D: RETURN CORRELATION MATRIX")
print("=" * 70)

ret_df = df[[f"{t}_ret" for t in TICKERS]].dropna()
ret_df.columns = [TICKER_NAMES[t] for t in TICKERS]
corr = ret_df.corr().round(4)
print(corr.to_string())
corr.to_csv(os.path.join(OUT_DIR, "tableD_return_correlation.csv"))
print(f"  → Saved to {OUT_DIR}/tableD_return_correlation.csv\n")

# =============================================================================
# TABLE E: Missing Data Summary
# =============================================================================
print("=" * 70)
print("  TABLE E: MISSING DATA SUMMARY")
print("=" * 70)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct,
    "Available": len(df) - missing,
}).sort_values("Missing %", ascending=False)
# Only show columns with missing data
missing_df = missing_df[missing_df["Missing Count"] > 0]
# Exclude columns that are intentionally empty (not used in analysis)
EXCLUDED_COLS = ["bnb_CapMVRVCur", "bnb_CapMrktCurUSD"]
missing_df = missing_df.drop(index=[c for c in EXCLUDED_COLS if c in missing_df.index], errors="ignore")
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("  No missing values detected in any column!")
missing_df.to_csv(os.path.join(OUT_DIR, "tableE_missing_data.csv"))
print(f"  → Saved to {OUT_DIR}/tableE_missing_data.csv\n")

# =============================================================================
# TABLE F: Full .describe() Export
# =============================================================================
print("=" * 70)
print("  TABLE F: FULL PANDAS DESCRIBE (all 99 columns)")
print("=" * 70)

full_desc = df.describe().T.round(6)
full_desc.to_csv(os.path.join(OUT_DIR, "tableF_full_describe.csv"))
print(f"  → Saved to {OUT_DIR}/tableF_full_describe.csv")
print(f"  (99 columns × 8 statistics = too large to print, see CSV)\n")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("  DESCRIPTIVE STATISTICS COMPLETE")
print("=" * 70)
print(f"  All {len(os.listdir(OUT_DIR))} tables saved to: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"    • {f}")
