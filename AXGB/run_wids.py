#!/usr/bin/env python3
"""Run all valid combinations on WIDS dataset (local CSV).
Uses feature grouping by null pattern similarity to handle the large number
of high-null features (175/183 columns have nulls).
"""
import itertools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

# --- Config ---
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NULL_THRESHOLD = 0.50   # Features with >50% nulls are candidates for extended
JACCARD_THRESHOLD = 0.95  # Group features with similar null patterns
MAX_GROUPS = 5           # Max number of groups to enumerate (2^5 = 32 combos)
MIN_GROUP_PCT = 0.02     # Min 2% of total population per test group

pipeline = GenericDataPipeline()

# --- Load data ---
csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'WIDS.csv')
df = pd.read_csv(csv_path, na_values=['NA'])

# Drop ID columns
columns_to_drop = ['encounter_id', 'patient_id', 'hospital_id']
df = df.drop(columns=columns_to_drop)
df = pipeline.preprocessing(df)

label = "hospital_death"
df[label] = df[label].astype(int)
print(f"Dataset shape (before feature selection): {df.shape}")

# --- Feature selection: keep top 50 by importance ---
import xgboost as xgb
print("Running quick XGBoost for feature selection...")
X_all = df.drop(label, axis=1)
selector = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              tree_method='hist', random_state=SEED, eval_metric='auc')
selector.fit(X_all, df[label], verbose=False)
imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)
top_features = set(imp.head(50).index.tolist())
keep_features = top_features | {label}
drop_cols = [c for c in df.columns if c not in keep_features]
df = df.drop(columns=drop_cols)
print(f"Feature selection: kept top {len(top_features)} features by importance")
print(f"Dropped {len(drop_cols)} features")

print(f"Dataset shape: {df.shape}")
print(f"N_TRIALS per model: {N_TRIALS}")
print(f"Target distribution:\n{df[label].value_counts()}")

# --- AXGB Feature Comparison ---
print(f"\n{'='*120}")
print("AXGB FEATURE COMPARISON")
print(f"{'='*120}")

from axgb.axgb_feature_comparison import AXGBFeatureComparison

# Define extended features (fill in the list with feature names)
extended_features = []  # TODO: Add extended feature names here

print(f"Extended features: {extended_features}")
print(f"Running AXGB comparison...")

# Initialize and run AXGB comparison
axgb_comparison = AXGBFeatureComparison(
    df=df.copy(),
    features_extended=extended_features,
    target_col=label,
    n_estimators=30,
    learning_rate=0.3,
    max_depth=6,
    max_window_size=1000,
    min_window_size=None,
    detect_drift=False,
    update_strategy='replace'
)

# Run the comparison
result = axgb_comparison.run()

print(f"\n{'='*120}")
print("AXGB SUMMARY")
print(f"{'='*120}")
print(f"AUC (Basic features only):  {result['auc_without_extended']:.4f}")
print(f"AUC (All features):         {result['auc_with_all']:.4f}")
print(f"Difference:                 {result['difference']:+.4f}")
print(f"{'='*120}")

print("\nAXGB Analysis Complete!")
