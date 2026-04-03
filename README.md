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
