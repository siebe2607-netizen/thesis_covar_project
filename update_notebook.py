import json

with open('ThesisMSc_FINAL.ipynb', 'r') as f:
    nb = json.load(f)

old_str1 = """from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg"""

new_str1 = """from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg"""

old_str2_a = """def estimate_forward_covar(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, test_size: float = 0.25,
    scale_features: bool = True
) -> dict:"""

new_str2_a = """def estimate_forward_covar(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, test_size: float = 0.25,
    scale_features: bool = True, use_quantreg: bool = True
) -> dict:"""

old_str2_b = """    mod  = QuantReg(y_tr, X_tr)
    res  = mod.fit(q=q, p_tol=1e-4, max_iter=2000)

    fitted   = res.predict(X_tr)
    forecast = res.predict(X_te)

    resid_full = y_tr - fitted
    base_mod   = QuantReg(y_tr, np.ones(len(y_tr)))
    base_res   = base_mod.fit(q=q, p_tol=1e-4, max_iter=2000)
    resid_base = y_tr - base_res.predict()

    def _check_loss(resid, q):
        return np.mean(resid * (q - (resid < 0).astype(float)))

    r2_is = 1 - _check_loss(resid_full, q) / _check_loss(resid_base, q)

    # ── Pinball (quantile) loss ──────────────────────────────────────────
    pinball_is  = _check_loss(y_tr - fitted,    q)
    pinball_oos = _check_loss(y_te - forecast,  q)

    # ── Coefficient table ────────────────────────────────────────────────
    param_names = ["const"] + feat_names
    params_df = pd.DataFrame(
        {
            "coef":   res.params,
            "pvalue": res.pvalues,
            "tstat":  res.tvalues,
        },
        index=param_names
    )

    return {
        "fitted":        pd.Series(fitted,   index=idx_tr, name="FwdCoVaR_fitted"),
        "forecast":      pd.Series(forecast, index=idx_te, name="FwdCoVaR_forecast"),
        "actual":        pd.Series(y_te,     index=idx_te, name="DeltaCoVaR_actual"),
        "params":        params_df,
        "pinball_is":    pinball_is,
        "pinball_oos":   pinball_oos,
        "r2_is":         r2_is,
        "feature_names": feat_names,
    }"""

new_str2_b = """    if use_quantreg:
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

    # ── Coefficient table ────────────────────────────────────────────────
    param_names = ["const"] + feat_names
    params_df = pd.DataFrame(
        {
            "coef":   res.params,
            "pvalue": res.pvalues,
            "tstat":  res.tvalues,
        },
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
    }"""

old_str3_a = """def estimate_forward_covar_expanding(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, min_train_size: int = 500,
    scale_features: bool = True
) -> dict:"""

new_str3_a = """def estimate_forward_covar_expanding(
    df: pd.DataFrame, ticker: str, delta_covar_series: pd.Series,
    q: float = QUANTILE, horizon: int = HORIZON, min_train_size: int = 500,
    scale_features: bool = True, use_quantreg: bool = True
) -> dict:"""

old_str3_b = """        # Fit Quantile Regression and Predict
        mod = QuantReg(y_train, X_tr)
        res = mod.fit(q=q, p_tol=1e-4, max_iter=2000)
        
        forecast = res.predict(X_te)[0]
        forecasts.append(forecast)
        forecast_idx.append(combined.index[i])

    # ── Final Reporting Model ──────────────────────────────────────────────
    # Refits on the entire dataset to generate the coefficients for your thesis tables
    if scale_features:
        full_mean = np.mean(Xm_arr, axis=0)
        full_std = np.std(Xm_arr, axis=0)
        full_std[full_std == 0] = 1.0
        Xm_arr_final = (Xm_arr - full_mean) / full_std
    else:
        Xm_arr_final = Xm_arr
    Xreg_full = add_constant(Xm_arr_final, has_constant="add")
    
    mod_full = QuantReg(y_arr, Xreg_full)
    res_full = mod_full.fit(q=q, p_tol=1e-4, max_iter=2000)
    fitted_full = res_full.predict(Xreg_full)
    
    # ── Loss Metrics (Pinball) ─────────────────────────────────────────────
    def _check_loss(resid, q_val):
        return np.mean(resid * (q_val - (resid < 0).astype(float)))

    resid_full = y_arr - fitted_full
    base_mod   = QuantReg(y_arr, np.ones(len(y_arr)))
    base_res   = base_mod.fit(q=q, p_tol=1e-4, max_iter=2000)
    resid_base = y_arr - base_res.predict()

    # R2 pseudo and In-sample loss
    r2_all = 1 - _check_loss(resid_full, q) / _check_loss(resid_base, q)
    pinball_is  = _check_loss(resid_full, q)
    
    # Out-of-sample loss on expanding window path
    y_test_actual = y_arr[min_train_size:]
    pinball_oos = _check_loss(y_test_actual - forecasts, q)

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
        "pinball_is":    pinball_is,
        "pinball_oos":   pinball_oos,
        "r2_is":         r2_all,
        "feature_names": feat_names,
    }"""

new_str3_b = """        # Fit Quantile Regression and Predict
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
    # Refits on the entire dataset to generate the coefficients for your thesis tables
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
    }"""

old_str4_a = """                      scale_features: bool = True,
                      use_expanding: bool = False,
                      verbose: bool = True) -> dict:"""

new_str4_a = """                      scale_features: bool = True,
                      use_expanding: bool = False,
                      use_quantreg: bool = True,
                      verbose: bool = True) -> dict:"""

old_str4_b = """                if use_expanding:
                    fwd = estimate_forward_covar_expanding(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features)
                else:
                    fwd = estimate_forward_covar(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features)

                res["forward"] = fwd
                if verbose:
                    print(f"  [3] Forward-CoVaR  |  Pseudo-R² = {fwd['r2_is']:.4f}"
                          f"  |  Pinball OOS = {fwd['pinball_oos']:.6f}")"""

new_str4_b = """                if use_expanding:
                    fwd = estimate_forward_covar_expanding(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features, use_quantreg=use_quantreg)
                else:
                    fwd = estimate_forward_covar(df, ticker, rolling_dcovar, q=q, horizon=horizon, scale_features=scale_features, use_quantreg=use_quantreg)

                res["forward"] = fwd
                if verbose:
                    print(f"  [3] Forward-CoVaR  |  Pseudo-R² = {fwd['r2_is']:.4f}"
                          f"  |  {fwd['loss_name']} OOS = {fwd['loss_oos']:.6f}")"""

old_str5 = """        row = {"Coin": t.upper(),
               "Pseudo-R²": round(fwd["r2_is"], 4),
               "Pinball IS": round(fwd["pinball_is"], 6),
               "Pinball OOS": round(fwd["pinball_oos"], 6)}"""

new_str5 = """        row = {"Coin": t.upper(),
               "Pseudo-R²": round(fwd["r2_is"], 4),
               f"{fwd['loss_name']} IS": round(fwd["loss_is"], 6),
               f"{fwd['loss_name']} OOS": round(fwd["loss_oos"], 6)}"""

old_str6 = """    # Show only fit stats + first 3 coefs to keep it readable
    show_cols = ["Pseudo-R²", "Pinball IS", "Pinball OOS"]"""

new_str6 = """    # Find dynamic loss names from the first available output
    loss_key_is = "Loss IS"
    loss_key_oos = "Loss OOS"
    if not fwd_tbl.empty:
        for col in fwd_tbl.columns:
            if " IS" in col:
                loss_key_is = col
            elif " OOS" in col:
                loss_key_oos = col

    show_cols = ["Pseudo-R²", loss_key_is, loss_key_oos]"""

old_str7_a = """                             horizons: list = [1, 5, 10], 
                             use_expanding: bool = False,
                             scale_features: bool = True,
                             verbose: bool = True) -> pd.DataFrame:"""

new_str7_a = """                             horizons: list = [1, 5, 10], 
                             use_expanding: bool = False,
                             scale_features: bool = True,
                             use_quantreg: bool = True,
                             verbose: bool = True) -> pd.DataFrame:"""

old_str7_b = """                    if use_expanding:
                        fwd = estimate_forward_covar_expanding(df_features, t, rolling_dcovar, q=q, horizon=h, scale_features=scale_features)
                    else:
                        fwd = estimate_forward_covar(df_features, t, rolling_dcovar, q=q, horizon=h, scale_features=scale_features)
                        
                    results_list.append({
                        "Coin": t.upper(),
                        "Quantile": q,
                        "Horizon": h,
                        "Engine": "Expanding" if use_expanding else "Static",
                        "Pseudo-R2": fwd["r2_is"],
                        "Pinball_IS": fwd["pinball_is"],
                        "Pinball_OOS": fwd["pinball_oos"]
                    })"""

new_str7_b = """                    if use_expanding:
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
                    })"""

replacements = [
    (old_str1, new_str1),
    (old_str2_a, new_str2_a),
    (old_str2_b, new_str2_b),
    (old_str3_a, new_str3_a),
    (old_str3_b, new_str3_b),
    (old_str4_a, new_str4_a),
    (old_str4_b, new_str4_b),
    (old_str5, new_str5),
    (old_str6, new_str6),
    (old_str7_a, new_str7_a),
    (old_str7_b, new_str7_b)
]

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
        
        # Split source back into list of lines, preserving newlines
        lines = []
        for line in source.splitlines(keepends=True):
            lines.append(line)
        cell['source'] = lines

with open('ThesisMSc_FINAL.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated!")
