import json

with open('ThesisMSc_FINAL.ipynb', 'r') as f:
    nb = json.load(f)

old_f_1 = """results = run_full_pipeline(full_df, 
                            q=0.05, 
                            horizon=1, 
                            scale_features=True, 
                            use_expanding=True, # <-- Toggle here!
                            use_quantreg=True)  # <-- Set False for strict A&B (OLS) Replication!"""

new_f_1 = """# ==========================================================
# AVAILABLE ENGINE PARAMETERS:
# q (float): Tail-risk Quantile (e.g., 0.05 for 5% tail, 0.01 for 1% tail)
# horizon (int): Forward prediction horizon in days (e.g. 1, 5, 10)
# scale_features (bool): True = Z-score standardization
# use_expanding (bool): True = Rolling OOS forecast, False = Static in-sample
# use_quantreg (bool): True = Quantile Regression, False = OLS Replication
# ==========================================================
results = run_full_pipeline(
    full_df, 
    q=0.05, 
    horizon=1, 
    scale_features=True, 
    use_expanding=True,
    use_quantreg=True
)"""

old_f_2 = """results = run_full_pipeline(full_df)"""

new_f_2 = """# ==========================================================
# AVAILABLE ENGINE PARAMETERS:
# q (float): Tail-risk Quantile (e.g., 0.05 for 5% tail, 0.01 for 1% tail)
# horizon (int): Forward prediction horizon in days (e.g. 1, 5, 10)
# scale_features (bool): True = Z-score standardization
# use_expanding (bool): True = Rolling OOS forecast, False = Static in-sample
# use_quantreg (bool): True = Quantile Regression, False = OLS Replication
# ==========================================================
results = run_full_pipeline(
    full_df, 
    q=0.05, 
    horizon=1, 
    scale_features=True, 
    use_expanding=True,
    use_quantreg=True
)"""

old_s_1 = """sens_df = run_sensitivity_analysis(full_df, 
                                   quantiles=[0.01, 0.05, 0.10], 
                                   horizons=[1, 5, 10], 
                                   scale_features=True, 
                                   use_expanding=False,
                                   use_quantreg=True)"""

new_s_1 = """# ==========================================================
# AVAILABLE SENSITIVITY PARAMETERS:
# quantiles (list): Tail-risk Quantiles to sweep (e.g., [0.01, 0.05])
# horizons (list): Forward prediction horizons (e.g. [1, 5, 10])
# scale_features (bool): True = Z-score standardization
# use_expanding (bool): True = Rolling OOS forecast, False = Static in-sample
# use_quantreg (bool): True = Quantile Regression, False = OLS Replication
# ==========================================================
sens_df = run_sensitivity_analysis(
    full_df, 
    quantiles=[0.01, 0.05, 0.10], 
    horizons=[1, 5, 10], 
    scale_features=True, 
    use_expanding=False, # Set to True for the full rigorous 40-minute matrix!
    use_quantreg=True
)"""

old_s_2 = """sens_df = run_sensitivity_analysis(full_df, 
                                   quantiles=[0.01, 0.05, 0.10], 
                                   horizons=[1, 5, 10], 
                                   scale_features=True, 
                                   use_expanding=True,
                                   use_quantreg=True)"""

new_s_2 = """# ==========================================================
# AVAILABLE SENSITIVITY PARAMETERS:
# quantiles (list): Tail-risk Quantiles to sweep (e.g., [0.01, 0.05])
# horizons (list): Forward prediction horizons (e.g. [1, 5, 10])
# scale_features (bool): True = Z-score standardization
# use_expanding (bool): True = Rolling OOS forecast, False = Static in-sample
# use_quantreg (bool): True = Quantile Regression, False = OLS Replication
# ==========================================================
sens_df = run_sensitivity_analysis(
    full_df, 
    quantiles=[0.01, 0.05, 0.10], 
    horizons=[1, 5, 10], 
    scale_features=True, 
    use_expanding=True, # Triggers rigorous 40-minute matrix
    use_quantreg=True
)"""

replacements = [
    (old_f_1, new_f_1),
    (old_f_2, new_f_2),
    (old_s_1, new_s_1),
    (old_s_2, new_s_2)
]

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
        
        # Format the JSON structure back cleanly
        lines = []
        for line in source.splitlines(keepends=True):
            lines.append(line)
        cell['source'] = lines

with open('ThesisMSc_FINAL.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Finished formatting parameters in notebook.")
