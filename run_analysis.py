import pandas as pd
import sys
import os

from covar_engine import (
    run_full_pipeline, 
    print_summary_report, 
    run_sensitivity_analysis,
    plot_sensitivity_heatmap
)

# Configuration
# By default, checking the user's Downloads folder for the exported dataset
CSV_PATH = os.path.expanduser('~/Downloads/thesis_full_df_backup_final.csv')

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find the dataset at {CSV_PATH}")
        print("Please ensure you have exported 'thesis_full_df_backup_final.csv' from your Jupyter Notebook.")
        sys.exit(1)

    print(f"Loading dataframe from: {CSV_PATH}")
    # Load dataset assuming DatetimeIndex was correctly exported as the first column
    full_df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    
    print("\n" + "="*70)
    print("  PHASE 1: RUNNING FULL FORWARD-CoVaR PIPELINE")
    print("="*70)
    
    # Run pipeline identical to notebook defaults
    results = run_full_pipeline(
        full_df, 
        q=0.05, 
        horizon=1, 
        scale_features=True, 
        use_expanding=True,
        verbose=True
    )
    
    print_summary_report(results)
    
    print("\n" + "="*70)
    print("  PHASE 2: RUNNING SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Run sensitivity grid using the expanding window engine
    sens_df = run_sensitivity_analysis(
        full_df, 
        quantiles=[0.01, 0.05, 0.10], 
        horizons=[1, 5, 10], 
        scale_features=True, 
        use_expanding=True
    )
    
    print("\n--- Sensitivity Analysis Results ---")
    print(sens_df.to_string())
    
    # Optional: Plotting
    print("\nGenerating Sensitivity Heatmap plot. Close the plot window to finish...")
    try:
        plot_sensitivity_heatmap(sens_df, metric='Pinball_OOS')
    except Exception as e:
        print(f"Plotting failed (expected if running without a display): {e}")
        
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
