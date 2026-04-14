#!/usr/bin/env python3
"""Run all valid combinations on Credit Risk Scoring dataset (Natural Nulls).

Dataset: Local CSVs (data_devsample.csv + data_to_score.csv merged on SK_ID_CURR).
Target: TARGET (binary: 1 = default, 0 = no default).
Uses feature grouping by null pattern similarity to handle large number of high-null features.
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
START_FROM = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # 1-based combo index to start from
NULL_THRESHOLD = 0.40   # Features with >40% nulls are candidates for extended
JACCARD_THRESHOLD = 0.95  # Group features with similar null patterns
MAX_GROUPS = 4           # Max number of groups to enumerate (2^4 = 16 combos)
MIN_GROUP_PCT = 0.02     # Min 2% of total population per test group

pipeline = GenericDataPipeline()

# --- Custom preprocessing for credit risk ---
def credit_risk_preprocessing(df):
    """Replace ? with NaN, convert objects to categorical codes, bools to int."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace(['?', ''], np.nan)
            if df[col].isna().sum() < len(df):
                df[col] = pd.Categorical(df[col]).codes
                df[col] = df[col].replace(-1, np.nan)
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    return df

# --- Load data ---
data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CreditRisk')
data1 = pd.read_csv(os.path.join(data_dir, 'data_devsample.csv'))
data2 = pd.read_csv(os.path.join(data_dir, 'data_to_score.csv'))
df = pd.merge(data1, data2, on='SK_ID_CURR', how='inner')
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop _y suffix columns, special columns, and SK_ID_CURR
columns_to_remove = [col for col in df.columns if col.endswith('_y') or col in ['TIME_x', 'BASE_x', 'DAY_x', 'MONTH_x']]
df = df.drop(columns=columns_to_remove, errors='ignore')
df = df.drop(columns=['SK_ID_CURR'], errors='ignore')

# Apply custom preprocessing then generic pipeline
df = credit_risk_preprocessing(df)
df = pipeline.preprocessing(df)

label = "TARGET"
df[label] = df[label].astype(int)

print(f"Dataset shape (before feature selection): {df.shape}")

# --- Feature selection: keep top 60 by importance + all high-null features ---
import xgboost as xgb
print("Running quick XGBoost for feature selection...")
X_all = df.drop(label, axis=1)
y_all = df[label]
selector = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              tree_method='hist', random_state=SEED, eval_metric='auc')
selector.fit(X_all, y_all, verbose=False)
imp = pd.Series(selector.feature_importances_, index=X_all.columns).sort_values(ascending=False)

# Keep top 40 features by importance (these become base + some ext)
top_features = set(imp.head(40).index.tolist())
keep_features = top_features | {label}

# Drop the rest
drop_cols = [c for c in df.columns if c not in keep_features]
df = df.drop(columns=drop_cols)
print(f"Feature selection: kept top {len(top_features)} features by importance")
print(f"Dropped {len(drop_cols)} features")
print(f"Dataset shape: {df.shape}")
print(f"N_TRIALS per model: {N_TRIALS}")
print(f"Target distribution:\n{df[label].value_counts()}")

# --- GBDT-IL Feature Comparison ---
print(f"\n{'='*120}")
print("GBDT-IL FEATURE COMPARISON")
print(f"{'='*120}")

from GBDTIL.gbdtil_feature_comparison import GBDTILFeatureComparison

# Define extended features (fill in the list with feature names)
extended_features = []  # TODO: Add extended feature names here

print(f"Extended features: {extended_features}")
print(f"Running GBDT-IL comparison...")

# Initialize and run GBDT-IL comparison
gbdtil_comparison = GBDTILFeatureComparison(
    df=df.copy(),
    features_extended=extended_features,
    target_col=label,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    max_window_size=50,
    subsample=1.0,
    
    
)

# Run the comparison
result = gbdtil_comparison.run()

print(f"\n{'='*120}")
print("GBDT-IL SUMMARY")
print(f"{'='*120}")
print(f"AUC (Basic features only):  {result['auc_without_extended']:.4f}")
print(f"AUC (All features):         {result['auc_with_all']:.4f}")
print(f"Difference:                 {result['difference']:+.4f}")
print(f"{'='*120}")

print("\nGBDT-IL Analysis Complete!")

