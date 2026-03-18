"""
run_all.py — Master runner: executes all 5 phases in sequence.

Usage:
    1. Download dataset from Kaggle:
       https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    2. Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder
    3. Install dependencies:
       pip install pandas numpy scikit-learn matplotlib seaborn joblib
    4. Run:
       python run_all.py
"""

import subprocess
import sys
import os
import time

phases = [
    ("Phase 1 — EDA",                          "phase1_eda.py"),
    ("Phase 2 — Feature Engineering",          "phase2_feature_engineering.py"),
    ("Phase 3 — Model Training & Comparison",  "phase3_model_training.py"),
    ("Phase 4 — Automated Pipeline",           "phase4_pipeline.py"),
    ("Phase 5 — Power BI Dashboard",           "phase5_powerbi_dashboard.py"),
]

print("=" * 60)
print("  ML PIPELINE — FULL RUN")
print("=" * 60)

# Check dataset exists
if not os.path.exists('WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    print("\n✗ Dataset not found!")
    print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("  Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in this folder.")
    sys.exit(1)

total_start = time.time()

for title, script in phases:
    print(f"\n{'─'*60}")
    print(f"  Running: {title}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  ✓ Completed in {elapsed:.1f}s")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"  ALL PHASES COMPLETE in {total:.1f}s")
print(f"  Ready to open predictions_powerbi.csv in Power BI!")
print(f"{'='*60}")