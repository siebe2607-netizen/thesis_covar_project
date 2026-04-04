#!/bin/bash
echo "====================================================="
echo "   BEGINNING AUTOMATED THESIS CoVaR BATCH PIPELINE   "
echo "====================================================="
echo ""
echo "[1/4] Running Main Analysis (Expand Window Quantiles)..."
python3 run_analysis.py
echo ""
echo "[2/4] Running Macroeconomic Stress Test Sandbox..."
python3 experiments/sandbox_stress_test.py
echo ""
echo "[3/4] Running Regime-Switching Dynamics Sandbox..."
python3 experiments/sandbox_regime_switching.py
echo ""
echo "[4/4] Running Extreme Value Theory (EVT) Sandbox..."
python3 experiments/sandbox_evt_analysis.py
echo ""
echo "====================================================="
echo "   ALL PIPELINES COMPLETE AND DATA EXPORTED TO CSV!  "
echo "====================================================="
