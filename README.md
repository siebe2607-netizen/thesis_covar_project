# Measuring Systemic Risk in the Cryptocurrency Market
### MSc Thesis — Forward-CoVaR Framework

This repository contains the complete analytical pipeline for estimating systemic risk contributions of major cryptocurrencies using the Forward-CoVaR framework (Adrian & Brunnermeier, 2016).

**Assets analysed:** Bitcoin (BTC), Ethereum (ETH), Ripple (XRP), Binance Coin (BNB), Solana (SOL)

---

## Repository Structure

```
thesis_covar_project/
├── ThesisMSc_FINAL.ipynb          # Primary Jupyter notebook (data fetching, cleaning, exploration)
├── covar_engine.py                # Standalone analytical engine (all regressions, backtests, tables)
├── run_analysis.py                # CLI runner — executes the full pipeline and exports results
├── thesis_full_df_backup_final.csv # Merged dataset (ready for analysis)
├── MsC_Thesis_Siebe-7.pdf        # Current thesis document
├── requirements.txt               # Python dependencies
├── results/                       # Generated output tables
│   ├── table1_unconditional_ranking.csv
│   ├── table2_conditional_ranking.csv
│   ├── table3_forward_covar.csv
│   ├── table4_backtests.csv
│   └── sensitivity_analysis.csv
└── experiments/                   # Experimental extensions (sandbox)
    ├── experimental_sandbox.ipynb
    ├── sandbox_evt_analysis.py    # Extreme Value Theory (Peaks-Over-Threshold)
    ├── sandbox_regime_switching.py # Markov regime-switching risk
    └── sandbox_stress_test.py     # Historical stress testing
```

## Methodology

The pipeline implements the three-stage CoVaR framework from Adrian & Brunnermeier (2016):

| Stage | Method | Description |
|-------|--------|-------------|
| **1. Unconditional CoVaR** | Quantile Regression | Static VaR and CoVaR estimation (A&B eq. 1–3) |
| **2. Conditional CoVaR** | Quantile Regression | Time-varying VaR and ΔCoVaR conditioned on market state M_{t-1} (A&B eq. 4–6) |
| **3. Forward-CoVaR** | OLS (default) or Quantile Regression | Predictive regression of ΔCoVaR on lagged market state and coin characteristics (A&B eq. 7) |

**Backtesting:** Kupiec (1995) Proportion of Failures and Christoffersen (1998) Independence tests.

### OLS vs Quantile Regression Toggle

The Forward-CoVaR stage supports both estimation methods via the `use_quantreg` parameter:

- **`use_quantreg=False` (Strict A&B replication):** Uses OLS to predict the conditional mean of ΔCoVaR. This matches the original Adrian & Brunnermeier (2016) specification, where stages 1–2 extract the tail risk measure via quantile regression, and stage 3 predicts its level via OLS. Reports standard R² and MSE.

- **`use_quantreg=True` (Extension):** Uses quantile regression at the specified tail quantile for the predictive step as well. This asks "how bad could ΔCoVaR get?" rather than "what is the expected ΔCoVaR?" Reports Pseudo-R² and Pinball loss.

### Estimation Windows

Two out-of-sample evaluation strategies are available via `use_expanding`:

- **Static split** (`use_expanding=False`): Single 75/25 train/test split.
- **Expanding window** (`use_expanding=True`): Re-trains daily on all available history up to time t, then predicts t+1. Methodologically stronger for genuine out-of-sample evaluation.

## Data Sources

The primary notebook (`ThesisMSc_FINAL.ipynb`) fetches and merges data from:

- **CoinMetrics** — on-chain fundamentals (active addresses, hashrate, MVRV)
- **DefiLlama** — DeFi TVL (Total Value Locked)
- **Artemis** — network activity metrics
- **CoinGecko** — market data (prices, market cap, volume)
- **Proprietary CSVs** — open interest and funding rate data

## Quick Start

### Option 1: Run from the pre-built dataset

```bash
git clone https://github.com/siebe2607-netizen/thesis_covar_project.git
cd thesis_covar_project
pip install -r requirements.txt
python3 run_analysis.py
```

This uses the included `thesis_full_df_backup_final.csv` and exports all result tables to `results/`.

### Option 2: Reproduce from scratch

1. Open `ThesisMSc_FINAL.ipynb` in Jupyter and run all cells to fetch data and build the merged dataset.
2. The notebook exports the dataset to CSV.
3. Run `python3 run_analysis.py` to generate results.

### Configuration

Edit the top of `run_analysis.py` to control the pipeline:

```python
USE_QUANTREG = False   # True = Quantile Regression extension, False = Strict A&B OLS
```

Or use `covar_engine.py` directly in Python:

```python
from covar_engine import run_full_pipeline, print_summary_report

results = run_full_pipeline(
    full_df,
    q=0.05,              # Tail quantile (5%)
    horizon=1,            # Forecast horizon (days)
    scale_features=True,  # Z-score standardisation
    use_expanding=True,   # Expanding window OOS evaluation
    use_quantreg=False    # OLS for Forward-CoVaR (A&B replication)
)
print_summary_report(results)
```

## Key Findings

- **SOL and ETH** emerge as the largest systemic risk contributors across both unconditional and conditional specifications.
- **XRP** consistently shows no improvement over the baseline model in the Forward-CoVaR prediction — a substantive negative finding indicating its systemic risk contribution is not forecastable from the current feature set.
- **SOL** exhibits a Kupiec backtest failure with violations below expected levels, suggesting the model is overly conservative for this asset. This is noted as a calibration limitation.
- **Leverage intensity** (OI/MC) and **macro volatility** (M_vol) are the most consistently significant predictors of forward-looking systemic risk.

## Sensitivity Analysis

The pipeline includes a sensitivity grid across:

- **Tail quantiles:** q = {0.01, 0.05, 0.10}
- **Forecast horizons:** h = {1, 5, 10} days

Results are exported to `results/sensitivity_analysis.csv` and can be visualised as heatmaps.

## Experimental Extensions

The `experiments/` folder contains sandbox implementations that do not modify the core engine:

- **Extreme Value Theory (EVT):** Models the 0.1% tail using Generalised Pareto Distribution (Peaks-Over-Threshold).
- **Regime-Switching Risk:** Markov-Switching models to evaluate whether leverage becomes deadlier under high-volatility regimes.
- **Historical Stress Testing:** Artificially shocks market state variables to simulate doomsday scenarios.

## References

- Adrian, T. & Brunnermeier, M.K. (2016). CoVaR. *American Economic Review*, 106(7), 1705–1741.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.
