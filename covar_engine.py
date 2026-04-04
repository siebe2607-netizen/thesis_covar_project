# =============================================================================
# FORWARD-CoVaR QUANTILE REGRESSION ENGINE
# MsC Thesis: Measuring Systemic Risk in the Cryptocurrency Market
# =============================================================================
# Implements:
#   1. Unconditional CoVaR  (Adrian & Brunnermeier 2016, eq. 1-3)
#   2. Conditional CoVaR    (A&B 2016, eq. 4-6)
#   3. Forward-CoVaR        (A&B 2016, eq. 7)
#   4. Backtesting          (Kupiec POF + Christoffersen Independence)
#   5. Pinball loss         (out-of-sample forecast evaluation)
#   6. Results tables + visualisations
#
# USAGE:  append this file to your existing notebook / run after full_df is built
# All column names match those produced by thesismsc.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

# ── colour palette (consistent across all plots) ────────────────────────────
COLORS = {
    "btc": "#F7931A", "eth": "#627EEA",
    "xrp": "#346AA9", "bnb": "#F3BA2F", "sol": "#9945FF"
}

TICKERS  = ["btc", "eth", "xrp", "bnb", "sol"]
QUANTILE = 0.05          # tail quantile (5 % left tail)
HORIZON  = 1             # Forward-CoVaR forecast horizon (days)


# =============================================================================
# SECTION 0 – FEATURE CONSTRUCTION
# =============================================================================
# Builds the derived variables that may not yet exist in full_df.
# Safe to re-run; existing columns are overwritten.

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs all coin-characteristic variables required for the regressions.
    Expects the column naming conventions produced by thesismsc.py.
    """
    df = df.copy()

    for t in TICKERS:
        # ── Log returns (idempotent) ──────────────────────────────────────
        price_col = f"{t}_PriceUSD"
        if price_col in df.columns:
            df[f"{t}_ret"] = np.log(df[price_col]).diff()

        # ── Leverage intensity: OI / MC ───────────────────────────────────
        oi_col = f"{t}_OI"
        mc_col = f"{t}_MC"
        if oi_col in df.columns and mc_col in df.columns:
            df[f"{t}_fac_lev"] = df[oi_col] / df[mc_col].replace(0, np.nan)

        # ── Network vitality: Δ ln(active addresses) ──────────────────────
        aa_col = f"{t}_AdrActCnt"
        if aa_col in df.columns:
            df[f"{t}_fac_netact"] = np.log(df[aa_col].replace(0, np.nan)).diff()

        # ── DeFi liquidity: Δ ln(TVL)  (ETH, BNB, SOL only) ─────────────
        tvl_col = f"{t}_TVL"
        if tvl_col in df.columns:
            df[f"{t}_fac_liq"] = np.log(df[tvl_col].replace(0, np.nan)).diff()

        # ── Valuation (ledger: BTC, XRP; DeFi: SOL, BNB; ETH both) ──────
        mvrv_col = f"{t}_CapMVRVCur"
        if mvrv_col in df.columns and f"{t}_fac_val" not in df.columns:
            df[f"{t}_fac_val"] = df[mvrv_col] - 1

        # MC/TVL fallback for DeFi assets
        if f"{t}_fac_val" not in df.columns:
            if mc_col in df.columns and tvl_col in df.columns:
                df[f"{t}_fac_val"] = (
                    df[mc_col] / df[tvl_col].replace(0, np.nan)
                )

    # ── PoW security: Δ ln(hashrate) × PoW dummy ────────────────────────
    if "btc_HashRate" in df.columns:
        df["btc_fac_security"] = np.log(
            df["btc_HashRate"].replace(0, np.nan)
        ).diff()

    # ETH PoW dummy already handled in thesismsc.py; replicate if absent
    if "eth_fac_security" not in df.columns and "eth_HashRate" in df.columns:
        eth_pow = (df.index < "2022-09-15").astype(float)
        df["eth_fac_security"] = (
            np.log(df["eth_HashRate"].replace(0, np.nan)).diff() * eth_pow
        )

    return df


def get_coin_chars(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Returns the coin-characteristic matrix X_i for a given ticker.
    Matches the exact column names constructed in your main thesis dataframe.
    """
    candidate_cols = {
        "lev":      f"{ticker}_Lev",
        "netact":   f"{ticker}_NetAct",
        "val":      f"{ticker}_Val",
        "liq":      f"{ticker}_Liq",
        "security": f"{ticker}_fac_security",
    }
    # Only keep columns that actually exist in the dataframe and aren't entirely NaNs
    present = {
        k: v for k, v in candidate_cols.items()
        if v in df.columns and not df[v].isna().all()
    }
    return df[list(present.values())].copy()


def get_market_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the market-state matrix M_{t-1}.
    Drops any column that is fully missing.
    """
    m_cols = ["M_vol", "M_trend", "M_fund", "M_fiat", "M_volu"]
    present = [c for c in m_cols if c in df.columns and not df[c].isna().all()]
    if not present:
        raise ValueError(
            "No market state variables found. "
            "Check that M_vol / M_trend / M_fiat / M_volu are built "
            "and that M_fund (funding rate) is merged into full_df."
        )
    return df[present].copy()


# =============================================================================
# SECTION 1 – UNCONDITIONAL CoVaR
# =============================================================================

def estimate_unconditional_covar(
    df: pd.DataFrame,
    ticker: str,
    q: float = QUANTILE
) -> pd.DataFrame:
    """
    Estimates static (unconditional) VaR and CoVaR via quantile regression.

    Step 1  –  VaR_i  :  r_i  = α_q  +  ε_q          (eq. 1)
    Step 2  –  CoVaR  :  r_sys = α_q^{sys|i}
                               + β_q^{sys|i} * r_i  +  ε_q    (eq. 2)
    Step 3  –  ΔCoVaR :  CoVaR(distress) − CoVaR(median)      (eq. 3)

    Returns a DataFrame indexed like df with columns:
        VaR_i, CoVaR_distress, CoVaR_median, DeltaCoVaR
    """
    ret_i   = df[f"{ticker}_ret"].dropna()
    ret_sys = df[f"sys_ret_excl_{ticker}"].reindex(ret_i.index).dropna()
    idx     = ret_i.index.intersection(ret_sys.index)

    ri  = ret_i.loc[idx].values
    rs  = ret_sys.loc[idx].values

    # ── Step 1: unconditional VaR at quantile q ──────────────────────────
    mod_var = QuantReg(ri, np.ones(len(ri)))
    res_var = mod_var.fit(q=q, vcov="iid")
    VaR_q   = float(res_var.params[0])          # scalar (constant model)

    res_med = mod_var.fit(q=0.5, vcov="iid")
    VaR_50  = float(res_med.params[0])

    # ── Step 2: CoVaR at quantile q ──────────────────────────────────────
    X       = add_constant(ri, has_constant="add")
    mod_cov = QuantReg(rs, X)
    res_cov = mod_cov.fit(q=q, vcov="iid")
    a_q, b_q = res_cov.params                   # intercept, slope

    # ── Step 3: ΔCoVaR ───────────────────────────────────────────────────
    CoVaR_distress = a_q + b_q * VaR_q
    CoVaR_median   = a_q + b_q * VaR_50
    DeltaCoVaR     = CoVaR_distress - CoVaR_median

    out = pd.DataFrame(
        {
            "VaR_i":           VaR_q,
            "CoVaR_distress":  CoVaR_distress,
            "CoVaR_median":    CoVaR_median,
            "DeltaCoVaR":      DeltaCoVaR,
        },
        index=idx
    )
    return out


# =============================================================================
# SECTION 2 – CONDITIONAL (TIME-VARYING) CoVaR
# =============================================================================

def estimate_conditional_covar(
    df: pd.DataFrame,
    ticker: str,
    q: float = QUANTILE
) -> pd.DataFrame:
    """
    Estimates time-varying VaR and CoVaR conditioned on market state M_{t-1}.

    Step 1  –  r_i_t  = α_q + β_q' M_{t-1}  +  ε_q             (eq. 4)
    Step 2  –  r_sys  = α_q^{sys|i} + β_q^{sys|i}' M_{t-1}
                      + γ_q^{sys|i} r_i  +  ε_q                 (eq. 5)
    Step 3  –  ΔCoVaR_t  =  CoVaR(r_i=VaR_q,t) − CoVaR(r_i=VaR_50,t) (eq. 6)

    Returns a DataFrame with columns:
        VaR_q_t, VaR_50_t, DeltaCoVaR_t
    plus coefficient DataFrames for both regressions (stored as attributes).
    """
    M   = get_market_state(df).shift(1)          # lag by 1 day
    ri  = df[f"{ticker}_ret"]
    rs  = df[f"sys_ret_excl_{ticker}"]

    # Align all series on a common clean index
    combined = pd.concat([ri, rs, M], axis=1).dropna()
    ri_  = combined.iloc[:, 0].values
    rs_  = combined.iloc[:, 1].values
    M_   = combined.iloc[:, 2:].values
    idx  = combined.index

    n_m  = M_.shape[1]

    # ── Step 1: VaR_i conditional on M ──────────────────────────────────
    X_var    = np.column_stack([np.ones(len(ri_)), M_])    # [1, M]
    mod_vq   = QuantReg(ri_, X_var)
    res_vq   = mod_vq.fit(q=q,   vcov="iid")
    res_v50  = mod_vq.fit(q=0.5, vcov="iid")

    VaR_q_t  = X_var @ res_vq.params               # fitted quantile path
    VaR_50_t = X_var @ res_v50.params

    # ── Step 2: CoVaR conditional on M and r_i ──────────────────────────
    # Design matrix: [1, M, r_i]
    X_cov    = np.column_stack([np.ones(len(rs_)), M_, ri_])
    mod_cov  = QuantReg(rs_, X_cov)
    res_cov  = mod_cov.fit(q=q, vcov="iid")

    # α, β (M coefficients), γ (coin return coefficient)
    alpha_q  = res_cov.params[0]
    beta_q   = res_cov.params[1:1 + n_m]
    gamma_q  = res_cov.params[-1]

    # ── Step 3: ΔCoVaR_t ────────────────────────────────────────────────
    # CoVaR at distress:  α + β'M + γ * VaR_q_t
    # CoVaR at median:    α + β'M + γ * VaR_50_t
    M_part        = M_ @ beta_q
    CoVaR_dist_t  = alpha_q + M_part + gamma_q * VaR_q_t
    CoVaR_med_t   = alpha_q + M_part + gamma_q * VaR_50_t
    DeltaCoVaR_t  = CoVaR_dist_t - CoVaR_med_t

    result = pd.DataFrame(
        {
            "VaR_q_t":      VaR_q_t,
            "VaR_50_t":     VaR_50_t,
            "CoVaR_dist_t": CoVaR_dist_t,
            "CoVaR_med_t":  CoVaR_med_t,
            "DeltaCoVaR_t": DeltaCoVaR_t,
        },
        index=idx
    )

    # Attach regression summaries as attributes for later inspection
    result.attrs["var_params"]  = pd.Series(res_vq.params,
                                            index=["const"] + list(combined.columns[2:]))
    result.attrs["covar_params"] = pd.Series(
        res_cov.params,
        index=["const"] + list(combined.columns[2:]) + [f"{ticker}_ret"]
    )
    result.attrs["var_pvalues"]   = res_vq.pvalues
    result.attrs["covar_pvalues"] = res_cov.pvalues

    return result


# =============================================================================
# SECTION 3 – FORWARD-CoVaR  (predictive regression, eq. 7)
# =============================================================================
def estimate_rolling_delta_covar(
    df: pd.DataFrame, ticker: str, q: float = QUANTILE, window: int = 250
) -> pd.Series:
    """
    Estimates time-varying ΔCoVaR using a rolling window.
    Safely rewritten using Pandas to perfectly match statsmodels alignments.
    """
    ri  = df[f"{ticker}_ret"]
    rs  = df[f"sys_ret_excl_{ticker}"]
    M   = get_market_state(df)

    # Drop NaNs to ensure a contiguous rolling window without gaps
    combined = pd.concat([ri, rs, M], axis=1).dropna()

    delta_vals = []
    delta_idx  = []

    macro_cols = list(M.columns)
    asset_name = ri.name

    for i in range(window, len(combined)):
        win = combined.iloc[i - window : i]

        y_asset = win[asset_name]
        y_sys   = win.iloc[:, 1]
        X_macro = add_constant(win[macro_cols])

        # Construct prediction row for time t
        pred_X = combined.iloc[i:i+1][macro_cols].copy()
        pred_X.insert(0, 'const', 1.0)

        try:
            # 1. Asset VaR Regressions
            model_vq  = QuantReg(y_asset, X_macro).fit(q=q, p_tol=1e-4, max_iter=2000)
            model_v50 = QuantReg(y_asset, X_macro).fit(q=0.50, p_tol=1e-4, max_iter=2000)

            var_q  = model_vq.predict(pred_X).iloc[0]
            var_50 = model_v50.predict(pred_X).iloc[0]

            # 2. System CoVaR Regression
            X_sys = pd.concat([y_asset, win[macro_cols]], axis=1)
            X_sys = add_constant(X_sys)

            model_covar = QuantReg(y_sys, X_sys).fit(q=q, p_tol=1e-4, max_iter=2000)

            # Extract gamma (the coefficient for the asset return)
            gamma_q = model_covar.params[asset_name]

            # Compute Delta CoVaR
            delta_vals.append(gamma_q * (var_q - var_50))
            delta_idx.append(combined.index[i])

        except Exception as e:
            # If it fails, append NaN so it doesn't break the time-series sync
            delta_vals.append(np.nan)
            delta_idx.append(combined.index[i])

    return pd.Series(delta_vals, index=delta_idx, name=f"DeltaCoVaR_rolling_{ticker}")


def estimate_forward_covar(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, test_size: float = 0.25,
    scale_features: bool = True, use_quantreg: bool = True
) -> dict:

    # 1. Catch silent rolling failures instantly
    if delta_covar_series.isna().all():
        raise ValueError("Rolling regressions failed entirely (returned all NaNs).")

    M  = get_market_state(df).shift(horizon)
    X  = get_coin_chars(df, ticker).shift(horizon)
    y  = delta_covar_series.rename("DeltaCoVaR")

    combined = pd.concat([y, M, X], axis=1).dropna()

    # 2. Catch empty dataframes before triggering the NumPy crash
    if len(combined) < 50:
        raise ValueError(f"Not enough data after aligning/dropping NaNs (found {len(combined)} rows). Check for missing coin characteristics.")

    y_arr  = combined["DeltaCoVaR"].values
    Xm_arr = combined.drop(columns=["DeltaCoVaR"]).values
    feat_names = list(combined.drop(columns=["DeltaCoVaR"]).columns)

    split   = int(len(y_arr) * (1 - test_size))
    X_tr_raw = Xm_arr[:split]
    X_te_raw = Xm_arr[split:]
    
    if scale_features:
        # --- Z-Score Standardization to maintain valid Feature Importance ---
        train_mean = np.mean(X_tr_raw, axis=0)
        train_std = np.std(X_tr_raw, axis=0)
        train_std[train_std == 0] = 1.0 # prevent div-by-zero
        
        X_tr_scaled = (X_tr_raw - train_mean) / train_std
        X_te_scaled = (X_te_raw - train_mean) / train_std
        
        X_tr = add_constant(X_tr_scaled, has_constant="add")
        X_te = add_constant(X_te_scaled, has_constant="add")
    else:
        X_tr = add_constant(X_tr_raw, has_constant="add")
        X_te = add_constant(X_te_raw, has_constant="add")
    y_tr, y_te = y_arr[:split],  y_arr[split:]
    idx_tr     = combined.index[:split]
    idx_te     = combined.index[split:]

    if use_quantreg:
        mod  = QuantReg(y_tr, X_tr)
        res  = mod.fit(q=q, p_tol=1e-4, max_iter=2000)
    else:
        mod  = sm.OLS(y_tr, X_tr)
        res  = mod.fit()

    fitted   = res.predict(X_tr)
    forecast = res.predict(X_te)

    resid_full = y_tr - fitted
    
    if use_quantreg:
        base_mod   = QuantReg(y_tr, np.ones(len(y_tr)))
        base_res   = base_mod.fit(q=q, p_tol=1e-4, max_iter=2000)
        resid_base = y_tr - base_res.predict()

        def _check_loss(resid, q_val):
            return np.mean(resid * (q_val - (resid < 0).astype(float)))

        r2_is = 1 - _check_loss(resid_full, q) / _check_loss(resid_base, q)
        loss_is  = _check_loss(y_tr - fitted,    q)
        loss_oos = _check_loss(y_te - forecast,  q)
        loss_name = "Pinball"
    else:
        r2_is = res.rsquared
        loss_is  = np.mean((y_tr - fitted)**2)
        loss_oos = np.mean((y_te - forecast)**2)
        loss_name = "MSE"

    param_names = ["const"] + feat_names
    params_df = pd.DataFrame(
        {"coef": res.params, "pvalue": res.pvalues, "tstat": res.tvalues},
        index=param_names
    )

    return {
        "fitted":        pd.Series(fitted,   index=idx_tr, name="FwdCoVaR_fitted"),
        "forecast":      pd.Series(forecast, index=idx_te, name="FwdCoVaR_forecast"),
        "actual":        pd.Series(y_te,     index=idx_te, name="DeltaCoVaR_actual"),
        "params":        params_df,
        "loss_is":       loss_is,
        "loss_oos":      loss_oos,
        "loss_name":     loss_name,
        "r2_is":         r2_is,
        "feature_names": feat_names,
    }



def estimate_forward_covar_expanding(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, min_train_size: int = 500,
    scale_features: bool = True, use_quantreg: bool = True
) -> dict:
    """
    Estimates Forward-CoVaR using an Expanding Window approach for out-of-sample prediction.
    As per the thesis methodology, variables are handled for scale and stationarity via 
    log-differences prior to this step, so no additional Z-score standardization is 
    applied to preserve economic interpretability of the coefficients.
    """
    import numpy as np
    from statsmodels.regression.quantile_regression import QuantReg
    from statsmodels.tools import add_constant

    if delta_covar_series.isna().all():
        raise ValueError("Rolling regressions failed entirely (returned all NaNs).")

    # Shift predictors by the forecast horizon
    M  = get_market_state(df).shift(horizon)
    X  = get_coin_chars(df, ticker).shift(horizon)
    y  = delta_covar_series.rename("DeltaCoVaR")

    combined = pd.concat([y, M, X], axis=1).dropna()

    if len(combined) <= min_train_size:
        raise ValueError(f"Not enough data for expanding window with min_train_size={min_train_size} (found {len(combined)} rows).")

    y_arr  = combined["DeltaCoVaR"].values
    Xm_arr = combined.drop(columns=["DeltaCoVaR"]).values
    feat_names = list(combined.drop(columns=["DeltaCoVaR"]).columns)
    
    forecasts = []
    forecast_idx = []
    
    # ── Expanding Window OOS Forecasting ──────────────────────────────────
    # Re-trains the model every day to predict day T+1
    for i in range(min_train_size, len(combined)):
        # Train on [0 : i], predict on [i]
        X_train  = Xm_arr[:i]
        y_train  = y_arr[:i]
        X_test   = Xm_arr[i:i+1] # Single day test slice
        
        if scale_features:
            # --- Dynamic Z-Score Standardization ---
            tr_mean = np.mean(X_train, axis=0)
            tr_std  = np.std(X_train, axis=0)
            tr_std[tr_std == 0] = 1.0
            
            X_train_sc = (X_train - tr_mean) / tr_std
            X_test_sc  = (X_test - tr_mean) / tr_std
            
            X_tr = add_constant(X_train_sc, has_constant="add")
            X_te = add_constant(X_test_sc, has_constant="add")
        else:
            X_tr = add_constant(X_train, has_constant="add")
            X_te = add_constant(X_test, has_constant="add")
        
        # Fit and Predict
        if use_quantreg:
            mod = QuantReg(y_train, X_tr)
            res = mod.fit(q=q, p_tol=1e-4, max_iter=2000)
        else:
            mod = sm.OLS(y_train, X_tr)
            res = mod.fit()
        
        forecast = res.predict(X_te)[0]
        forecasts.append(forecast)
        forecast_idx.append(combined.index[i])

    # ── Final Reporting Model ──────────────────────────────────────────────
    if scale_features:
        full_mean = np.mean(Xm_arr, axis=0)
        full_std = np.std(Xm_arr, axis=0)
        full_std[full_std == 0] = 1.0
        Xm_arr_final = (Xm_arr - full_mean) / full_std
    else:
        Xm_arr_final = Xm_arr
    Xreg_full = add_constant(Xm_arr_final, has_constant="add")
    
    if use_quantreg:
        mod_full = QuantReg(y_arr, Xreg_full)
        res_full = mod_full.fit(q=q, p_tol=1e-4, max_iter=2000)
    else:
        mod_full = sm.OLS(y_arr, Xreg_full)
        res_full = mod_full.fit()

    fitted_full = res_full.predict(Xreg_full)
    
    # ── Loss Metrics ─────────────────────────────────────────────
    resid_full = y_arr - fitted_full
    y_test_actual = y_arr[min_train_size:]

    if use_quantreg:
        def _check_loss(resid, q_val):
            return np.mean(resid * (q_val - (resid < 0).astype(float)))

        base_mod   = QuantReg(y_arr, np.ones(len(y_arr)))
        base_res   = base_mod.fit(q=q, p_tol=1e-4, max_iter=2000)
        resid_base = y_arr - base_res.predict()

        r2_all = 1 - _check_loss(resid_full, q) / _check_loss(resid_base, q)
        loss_is  = _check_loss(resid_full, q)
        loss_oos = _check_loss(y_test_actual - forecasts, q)
        loss_name = "Pinball"
    else:
        r2_all = res_full.rsquared
        loss_is  = np.mean(resid_full ** 2)
        loss_oos = np.mean((y_test_actual - forecasts) ** 2)
        loss_name = "MSE"

    param_names = ["const"] + feat_names
    params_df = pd.DataFrame(
        {"coef": res_full.params, "pvalue": res_full.pvalues, "tstat": res_full.tvalues},
        index=param_names
    )

    return {
        "fitted":        pd.Series(fitted_full, index=combined.index, name="FwdCoVaR_fitted"),
        "forecast":      pd.Series(forecasts, index=forecast_idx, name="FwdCoVaR_forecast"),
        "actual":        pd.Series(y_test_actual, index=forecast_idx, name="DeltaCoVaR_actual"),
        "params":        params_df,
        "loss_is":       loss_is,
        "loss_oos":      loss_oos,
        "loss_name":     loss_name,
        "r2_is":         r2_all,
        "feature_names": feat_names,
    }

# =============================================================================
# SECTION 4 – BACKTESTING  (Kupiec + Christoffersen)
# =============================================================================

def kupiec_pof_test(returns: np.ndarray, var_forecasts: np.ndarray,
                    q: float = QUANTILE) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    H0: violation frequency = q  (correctly specified VaR)
    Returns LR statistic and p-value.
    """
    violations = (returns < var_forecasts).astype(int)
    T  = len(violations)
    N  = violations.sum()
    p  = N / T                                    # observed failure rate

    if p == 0 or p == 1:
        return {"N": N, "T": T, "p_hat": p, "LR": np.nan, "pvalue": np.nan,
                "reject_H0": None}

    LR = -2 * (
        np.log((1 - q) ** (T - N) * q ** N)
        - np.log((1 - p) ** (T - N) * p ** N)
    )
    pvalue = 1 - stats.chi2.cdf(LR, df=1)
    return {
        "N": N, "T": T, "p_hat": round(p, 4),
        "LR": round(LR, 4), "pvalue": round(pvalue, 4),
        "reject_H0": pvalue < 0.05
    }


def christoffersen_independence_test(returns: np.ndarray,
                                     var_forecasts: np.ndarray,
                                     q: float = QUANTILE) -> dict:
    """
    Christoffersen (1998) independence test.

    Tests whether VaR violations cluster (reject H0 = violations are i.i.d.).
    """
    hits = (returns < var_forecasts).astype(int)
    T    = len(hits)

    # Transition counts
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    # Transition probabilities
    pi_01 = n01 / (n00 + n01 + 1e-12)
    pi_11 = n11 / (n10 + n11 + 1e-12)
    pi    = (n01 + n11) / (n00 + n01 + n10 + n11 + 1e-12)

    # Likelihood ratio
    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    logL_uncond = (
        (n00 + n10) * _safe_log(1 - pi)
        + (n01 + n11) * _safe_log(pi)
    )
    logL_cond = (
        n00 * _safe_log(1 - pi_01) + n01 * _safe_log(pi_01)
        + n10 * _safe_log(1 - pi_11) + n11 * _safe_log(pi_11)
    )

    LR_ind = -2 * (logL_uncond - logL_cond)
    pvalue = 1 - stats.chi2.cdf(LR_ind, df=1)

    return {
        "pi_01": round(pi_01, 4), "pi_11": round(pi_11, 4),
        "LR_ind": round(LR_ind, 4), "pvalue": round(pvalue, 4),
        "reject_H0": pvalue < 0.05
    }


def run_backtests(df: pd.DataFrame, ticker: str,
                  cond_results: pd.DataFrame,
                  q: float = QUANTILE) -> dict:
    """
    Runs both Kupiec and Christoffersen tests on the conditional VaR forecasts.
    Uses the last 25 % of the sample as the evaluation window.
    """
    ret_col = f"{ticker}_ret"
    combined = pd.concat(
        [df[ret_col], cond_results["VaR_q_t"]], axis=1
    ).dropna()

    split   = int(len(combined) * 0.75)
    r_test  = combined.iloc[split:, 0].values
    var_test = combined.iloc[split:, 1].values

    kupiec  = kupiec_pof_test(r_test, var_test, q)
    christ  = christoffersen_independence_test(r_test, var_test, q)

    return {"kupiec": kupiec, "christoffersen": christ,
            "n_test": len(r_test)}


# =============================================================================
# SECTION 5 – FULL PIPELINE
# =============================================================================

def run_full_pipeline(df: pd.DataFrame,
                      q: float = QUANTILE,
                      horizon: int = HORIZON,
                      scale_features: bool = True,
                      use_expanding: bool = False,
                      use_quantreg: bool = True,
                      verbose: bool = True) -> dict:
    """
    Runs the complete Forward-CoVaR pipeline for all tickers and returns
    a results dictionary with all intermediate outputs.
    """
    df = build_features(df)

    results = {}

    for ticker in TICKERS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Processing: {ticker.upper()}")
            print(f"{'='*60}")

        res = {}

        # ── 1. Unconditional CoVaR ────────────────────────────────────────
        try:
            uncond = estimate_unconditional_covar(df, ticker, q)
            res["unconditional"] = uncond
            if verbose:
                print(f"  [1] Unconditional ΔCoVaR : {uncond['DeltaCoVaR'].mean():.4f}")
        except Exception as e:
            print(f"  [ERROR] Unconditional CoVaR for {ticker}: {e}")

        # ── 2. Conditional CoVaR ─────────────────────────────────────────
        try:
            cond = estimate_conditional_covar(df, ticker, q)
            res["conditional"] = cond
            if verbose:
                print(f"  [2] Conditional ΔCoVaR (mean): "
                      f"{cond['DeltaCoVaR_t'].mean():.4f}")
                print(f"      γ (coin return coeff):      "
                      f"{cond.attrs['covar_params'].iloc[-1]:.4f}")
        except Exception as e:
            print(f"  [ERROR] Conditional CoVaR for {ticker}: {e}")

        # ── 3. Forward-CoVaR ─────────────────────────────────────────────
        try:
            if "conditional" in res:
                rolling_dcovar = estimate_rolling_delta_covar(df, ticker, q=q)
                res["rolling_dcovar"] = rolling_dcovar

                if use_expanding:
                    fwd = estimate_forward_covar_expanding(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features, use_quantreg=use_quantreg)
                else:
                    fwd = estimate_forward_covar(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features, use_quantreg=use_quantreg)

                res["forward"] = fwd
                if verbose:
                    print(f"  [3] Forward-CoVaR  |  Pseudo-R² = {fwd['r2_is']:.4f}"
                          f"  |  {fwd['loss_name']} OOS = {fwd['loss_oos']:.6f}")
        except Exception as e:
            print(f"  [ERROR] Forward-CoVaR for {ticker}: {e}")

        # ── 4. Backtests ──────────────────────────────────────────────────
        try:
            if "conditional" in res:
                bt = run_backtests(df, ticker, res["conditional"], q)
                res["backtests"] = bt
                if verbose:
                    kp = bt["kupiec"]
                    ch = bt["christoffersen"]
                    print(f"  [4] Kupiec  : LR={kp['LR']}, p={kp['pvalue']}, "
                          f"reject={kp['reject_H0']}")
                    print(f"      Christ. : LR={ch['LR_ind']}, p={ch['pvalue']}, "
                          f"reject={ch['reject_H0']}")
        except Exception as e:
            print(f"  [ERROR] Backtests for {ticker}: {e}")

        results[ticker] = res

    return results


# =============================================================================
# SECTION 6 – RESULTS TABLES
# =============================================================================

def make_unconditional_ranking_table(results: dict) -> pd.DataFrame:
    """
    Table 1 in the thesis: static ΔCoVaR ranking across coins.
    """
    rows = []
    for t in TICKERS:
        if "unconditional" not in results.get(t, {}):
            continue
        unc = results[t]["unconditional"]
        rows.append({
            "Coin":       t.upper(),
            "VaR (5%)":   round(unc["VaR_i"].mean(), 4),
            "CoVaR (5%)": round(unc["CoVaR_distress"].mean(), 4),
            "ΔCoVaR":     round(unc["DeltaCoVaR"].mean(), 4),
        })
    df_rank = pd.DataFrame(rows).sort_values("ΔCoVaR")
    df_rank.insert(0, "Rank", range(1, len(df_rank) + 1))
    return df_rank.set_index("Rank")


def make_conditional_ranking_table(results: dict) -> pd.DataFrame:
    """
    Table 2: time-averaged conditional ΔCoVaR ranking.
    """
    rows = []
    for t in TICKERS:
        if "conditional" not in results.get(t, {}):
            continue
        cond = results[t]["conditional"]
        rows.append({
            "Coin":             t.upper(),
            "Mean ΔCoVaR_t":    round(cond["DeltaCoVaR_t"].mean(), 4),
            "Min ΔCoVaR_t":     round(cond["DeltaCoVaR_t"].min(),  4),
            "Max ΔCoVaR_t":     round(cond["DeltaCoVaR_t"].max(),  4),
        })
    df_rank = pd.DataFrame(rows).sort_values("Mean ΔCoVaR_t")
    df_rank.insert(0, "Rank", range(1, len(df_rank) + 1))
    return df_rank.set_index("Rank")


def make_forward_covar_table(results: dict) -> pd.DataFrame:
    """
    Table 3: Forward-CoVaR regression coefficients and fit statistics.
    """
    rows = []
    for t in TICKERS:
        if "forward" not in results.get(t, {}):
            continue
        fwd = results[t]["forward"]
        row = {"Coin": t.upper(),
               "Pseudo-R²": round(fwd["r2_is"], 4),
               f"{fwd['loss_name']} IS": round(fwd["loss_is"], 6),
               f"{fwd['loss_name']} OOS": round(fwd["loss_oos"], 6)}
        for feat, coef, pval in zip(
            fwd["params"].index,
            fwd["params"]["coef"],
            fwd["params"]["pvalue"]
        ):
            row[f"β_{feat}"]  = round(coef,  4)
            row[f"p_{feat}"]  = round(pval,  4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Coin")


def make_backtest_table(results: dict) -> pd.DataFrame:
    """
    Table 4: Backtest summary (Kupiec + Christoffersen).
    """
    rows = []
    for t in TICKERS:
        bt = results.get(t, {}).get("backtests")
        if bt is None:
            continue
        kp = bt["kupiec"]
        ch = bt["christoffersen"]
        rows.append({
            "Coin":        t.upper(),
            "N obs":       bt["n_test"],
            "Violations":  kp["N"],
            "Viol. rate":  kp["p_hat"],
            "Kupiec LR":   kp["LR"],
            "Kupiec p":    kp["pvalue"],
            "Kupiec pass": "✓" if not kp["reject_H0"] else "✗",
            "Christ. LR":  ch["LR_ind"],
            "Christ. p":   ch["pvalue"],
            "Christ. pass":"✓" if not ch["reject_H0"] else "✗",
        })
    return pd.DataFrame(rows).set_index("Coin")


# =============================================================================
# SECTION 7 – VISUALISATIONS
# =============================================================================

def plot_dynamic_covar(results: dict, save_path: str = None):
    """
    Figure 1: Time-varying ΔCoVaR for all coins on a single plot.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    for t in TICKERS:
        cond = results.get(t, {}).get("conditional")
        if cond is None:
            continue
        s = cond["DeltaCoVaR_t"]
        ax.plot(s.index, s.values, label=t.upper(),
                color=COLORS[t], linewidth=1.1, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Time-Varying ΔCoVaR – Systemic Risk Contributions",
                 fontsize=13)
    ax.set_ylabel("ΔCoVaR (daily log-return units)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_forward_covar_fit(results: dict, ticker: str, save_path: str = None):
    """
    Figure 2: Actual vs Forward-ΔCoVaR (in-sample + out-of-sample).
    """
    fwd  = results[ticker]["forward"]
    cond = results[ticker]["conditional"]

    actual_full = cond["DeltaCoVaR_t"]
    fitted      = fwd["fitted"]
    forecast    = fwd["forecast"]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(actual_full.index, actual_full.values,
            color="grey", linewidth=0.8, alpha=0.6, label="Realised ΔCoVaR")
    ax.plot(fitted.index, fitted.values,
            color=COLORS[ticker], linewidth=1.2, label="Fitted (in-sample)")
    ax.plot(forecast.index, forecast.values,
            color=COLORS[ticker], linewidth=1.5, linestyle="--",
            label="Forecast (out-of-sample)")

    # Shade OOS window
    ax.axvspan(forecast.index[0], forecast.index[-1],
               alpha=0.07, color=COLORS[ticker])
    ax.axvline(forecast.index[0], color="black",
               linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_title(f"{ticker.upper()} – Forward-ΔCoVaR vs Realised",
                 fontsize=13)
    ax.set_ylabel("ΔCoVaR")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_ranking_shift(results: dict, save_path: str = None):
    """
    Figure 3: Slope chart showing how systemic importance ranking
    changes from unconditional to conditional ΔCoVaR.
    """
    unc_means = {}
    cond_means = {}

    for t in TICKERS:
        unc  = results.get(t, {}).get("unconditional")
        cond = results.get(t, {}).get("conditional")
        if unc is not None:
            unc_means[t.upper()]  = unc["DeltaCoVaR"].mean()
        if cond is not None:
            cond_means[t.upper()] = cond["DeltaCoVaR_t"].mean()

    coins   = list(unc_means.keys())
    unc_v   = [unc_means[c]  for c in coins]
    cond_v  = [cond_means.get(c, np.nan) for c in coins]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, coin in enumerate(coins):
        color = COLORS[coin.lower()]
        ax.plot([0, 1], [unc_v[i], cond_v[i]],
                color=color, linewidth=2, marker="o", markersize=7)
        ax.text(-0.05, unc_v[i],  f"{coin}", ha="right", va="center",
                color=color, fontsize=10, fontweight="bold")
        ax.text(1.05,  cond_v[i], f"{coin}", ha="left",  va="center",
                color=color, fontsize=10, fontweight="bold")

    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Unconditional", "Conditional"], fontsize=11)
    ax.set_ylabel("Mean ΔCoVaR")
    ax.set_title("Systemic Importance Ranking Shift", fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(results: dict, save_path: str = None):
    """
    Figure 4: Coefficient bar chart from the Forward-CoVaR regressions,
    one panel per coin.  Only significant coefficients (p < 0.10) are solid.
    """
    tickers_with_fwd = [t for t in TICKERS
                        if "forward" in results.get(t, {})]
    n = len(tickers_with_fwd)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, tickers_with_fwd):
        params = results[t]["forward"]["params"].drop("const", errors="ignore")
        coefs  = params["coef"]
        pvals  = params["pvalue"]

        colors_bar = [
            COLORS[t] if p < 0.10 else "lightgrey"
            for p in pvals
        ]
        ax.barh(coefs.index, coefs.values, color=colors_bar, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(t.upper(), fontsize=11, color=COLORS[t],
                     fontweight="bold")
        ax.set_xlabel("Coefficient")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Forward-ΔCoVaR Regression Coefficients\n"
                 "(solid = p < 0.10)", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# SECTION 8 – PRINT SUMMARY REPORT
# =============================================================================

def print_summary_report(results: dict):
    """Prints all tables to console in a readable format."""

    sep = "=" * 70

    print(f"\n{sep}")
    print("  TABLE 1 – UNCONDITIONAL ΔCoVaR RANKING")
    print(sep)
    print(make_unconditional_ranking_table(results).to_string())

    print(f"\n{sep}")
    print("  TABLE 2 – CONDITIONAL ΔCoVaR RANKING")
    print(sep)
    print(make_conditional_ranking_table(results).to_string())

    print(f"\n{sep}")
    print("  TABLE 3 – FORWARD-CoVaR REGRESSION (in-sample)")
    print(sep)
    fwd_tbl = make_forward_covar_table(results)
    # Find dynamic loss names from the first available output
    loss_key_is = "Loss IS"
    loss_key_oos = "Loss OOS"
    if not fwd_tbl.empty:
        for col in fwd_tbl.columns:
            if " IS" in col:
                loss_key_is = col
            elif " OOS" in col:
                loss_key_oos = col

    show_cols = ["Pseudo-R²", loss_key_is, loss_key_oos]
    coef_cols = [c for c in fwd_tbl.columns if c.startswith("β_")][:6]
    print(fwd_tbl[show_cols + coef_cols].to_string())

    print(f"\n{sep}")
    print("  TABLE 4 – BACKTESTING SUMMARY")
    print(sep)
    print(make_backtest_table(results).to_string())


# =============================================================================
# ENTRY POINT
# =============================================================================
# In your notebook, after full_df is built, simply call:
#
#   from covar_engine import run_full_pipeline, print_summary_report
#   from covar_engine import plot_dynamic_covar, plot_forward_covar_fit
#   from covar_engine import plot_ranking_shift, plot_feature_importance
#
#   results = run_full_pipeline(full_df)
#   print_summary_report(results)
#   plot_dynamic_covar(results)
#   plot_ranking_shift(results)
#   plot_feature_importance(results)
#   for t in ["btc", "eth", "xrp", "bnb", "sol"]:
#       plot_forward_covar_fit(results, t)
#
# =============================================================================

# =============================================================================
# SECTION 9 – SENSITIVITY ANALYSIS (HORIZON & TAIL)
# =============================================================================

def run_sensitivity_analysis(df: pd.DataFrame, 
                             tickers: list = TICKERS, 
                             quantiles: list = [0.01, 0.05, 0.10], 
                             horizons: list = [1, 5, 10], 
                             use_expanding: bool = False,
                             scale_features: bool = True,
                             use_quantreg: bool = True,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Runs a sensitivity analysis across multiple tail quantiles (q) and prediction horizons.
    Returns a MultiIndex DataFrame with Pinball and R2 metrics.
    """
    df_features = build_features(df)
    results_list = []
    
    for t in tickers:
        if verbose:
            print(f"\n{'='*50}\n  Sensitivity Analysis: {t.upper()}\n{'='*50}")
            
        for q in quantiles:
            if verbose:
                print(f"  --> Calculating Conditional CoVaR for q={q}...")
                
            # DeltaCoVaR only depends on q, not horizon! Calculate once per quantile.
            rolling_dcovar = estimate_rolling_delta_covar(df_features, t, q=q)
            
            for h in horizons:
                if verbose:
                    print(f"      Running Forward-CoVaR (h={h}) | Expanding={use_expanding}")
                try:
                    if use_expanding:
                        fwd = estimate_forward_covar_expanding(df_features, t, rolling_dcovar, q=q, horizon=h, scale_features=scale_features, use_quantreg=use_quantreg)
                    else:
                        fwd = estimate_forward_covar(df_features, t, rolling_dcovar, q=q, horizon=h, scale_features=scale_features, use_quantreg=use_quantreg)
                        
                    results_list.append({
                        "Coin": t.upper(),
                        "Quantile": q,
                        "Horizon": h,
                        "Engine": "Expanding" if use_expanding else "Static",
                        "Pseudo-R2": fwd["r2_is"],
                        "Loss_IS": fwd["loss_is"],
                        "Loss_OOS": fwd["loss_oos"],
                        "Loss_Metric": fwd["loss_name"]
                    })
                except Exception as e:
                    print(f"      [ERROR] failed for q={q}, h={h}: {e}")
                    
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        results_df.set_index(["Coin", "Quantile", "Horizon"], inplace=True)
    return results_df


def plot_sensitivity_heatmap(sens_df: pd.DataFrame, metric: str = 'Pinball_OOS'):
    """
    Plots a heatmap of the Sensitivity Analysis results.
    requires: import seaborn as sns
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if sens_df.empty:
        print("No data to plot.")
        return
        
    coins = sens_df.index.get_level_values('Coin').unique()
    
    fig, axes = plt.subplots(1, len(coins), figsize=(5 * len(coins), 4), sharey=True)
    if len(coins) == 1:
        axes = [axes]
        
    for ax, coin in zip(axes, coins):
        subset = sens_df.xs(coin, level='Coin')
        # Handle index effectively
        pivot = subset.reset_index().pivot_table(index='Quantile', columns='Horizon', values=metric)
        
        sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".4f", ax=ax, cbar_kws={'label': metric})
        ax.set_title(f"{coin} - {metric}")
        ax.invert_yaxis()
        
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("covar_engine.py loaded successfully.")
