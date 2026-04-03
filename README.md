# Master's Thesis CoVaR Engine

This is the standalone Python repository housing the Forward-CoVaR analysis pipeline.

## Structure
- `covar_engine.py`: Defines the quantile regressions, model fitting, scaling methods, backtesting methods, and result processing. It has been perfectly transferred from the primary Jupyter notebook.
- `run_analysis.py`: Designed to execute the engine on your exported dataset safely and cleanly, directly outputting the metrics needed for your tables.
- `requirements.txt`: Project dependencies.

## Usage

1. Open your master Jupyter notebook, and run it locally so that it outputs your final merged dataset via:
   `full_df.to_csv('/Users/iphonevansiebe/Downloads/thesis_full_df_backup_final.csv', index=True)`
   
2. Open terminal and navigate to the project directory:
   ```bash
   cd /Users/iphonevansiebe/.gemini/antigravity/scratch/thesis_covar_project
   ```

3. Install requirements (optional but recommended in a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis:
   ```bash
   python3 run_analysis.py
   ```

## Future Extensions & Experimental Ideas

Now that the core mathematical engine is isolated from formatting code, several exciting analytical extensions can be safely built in the `experiments/` sandbox folder:

### 1. Directional Spillover Effects (Diebold-Yilmaz)
Currently, CoVaR traces how the *overall system* reacts when a specific coin is in distress. By implementing a Vector Autoregression (VAR) model and the Diebold-Yilmaz spillover index, we can trace directional contagion risk. For example: does a crash in Ethereum spill over into Solana more fiercely than a Solana crash spills into Ethereum?

### 2. Regime-Switching Risk (Bull vs. Bear Dynamics)
The current pipeline evaluates the macroeconomic states ($M_{t-1}$) linearly. However, market microstructures behave fundamentally differently in euphoric bull markets versus devastating bear markets. By incorporating a Markov-Switching model, we can evaluate whether variables like *leverage* become specifically deadlier under high-volatility distressed regimes compared to normal regimes.

### 3. Non-Linear Machine Learning with SHAP
Financial data is rarely perfectly linear. For example, high leverage may only trigger systemic liquidations *if* network activity simultaneously plummets. Swapping the linear Quantile Regression out for Quantile Random Forests (QRF) or Gradient Boosting Regression Trees allows us to capture these explosive non-linear interactions. Plotting SHAP values over time could dynamically map exactly when certain on-chain conditions reach a boiling point.

### 4. Extreme Value Theory (EVT)
The primary analytical pipeline operates at the 5% and 1% tail quantiles. Since crypto markets exhibit dramatic fat-tails, we can mathematically model the extreme edge of the distribution (the 0.1% tail) by swapping the regression engine for Extreme Value Theory (EVT) methods like Block Maxima or Peaks-Over-Threshold.

### 5. Historical Stress Testing
By isolating the predictive Forward-CoVaR engine, we can artificially shock the historical variables to simulate "Doomsday Scenarios". For example, artificially multiplying `M_vol` (Macro Volatility) by 400% inside a controlled sandbox allows us to forecast exactly how much monetary value the cryptocurrency ecosystem would bleed out tomorrow if a Black Swan event were to suddenly multiply market panic.
