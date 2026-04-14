#!/usr/bin/env python3
"""Run all valid combinations on Flight Delay Prediction dataset (Augmented).

Dataset: divyansh22/flight-delay-prediction (Kaggle) — Jan 2019 on-time data.
Target: DEP_DEL15 (binary: 1 = departure delayed ≥15 min).
Augmentation: inject nulls into selected features to simulate incremental data.
"""
import itertools
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import kagglehub
from core.GenericDataPipeline import GenericDataPipeline
from core.RunData import RunPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from core.seed_utils import SEED, set_all_seeds

pd.set_option('future.infer_string', False)
set_all_seeds()

# --- Config ---
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NULL_THRESHOLD = 0.05  # Features with >5% nulls are candidates for extended
MIN_GROUP_PCT = 0.02   # Min 2% of total population per test group

pipeline = GenericDataPipeline()

# --- Load data ---
path = kagglehub.dataset_download("divyansh22/flight-delay-prediction")
df = pd.read_csv(os.path.join(path, "Jan_2019_ontime.csv"))

# Drop leaky/useless columns
# ARR_TIME, ARR_DEL15: arrival info leaks future info for departure delay prediction
# CANCELLED, DIVERTED: post-hoc labels
# DEP_TIME: leaks exact departure time (directly determines delay)
# Unnamed: 21: empty column
# *_SEQ_ID, OP_CARRIER_AIRLINE_ID: redundant with other airport/carrier columns
drop_cols = ['Unnamed: 21', 'CANCELLED', 'DIVERTED', 'DEP_TIME', 'ARR_TIME',
             'ARR_DEL15', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID',
             'OP_CARRIER_AIRLINE_ID']
df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

# Drop rows with missing target
df.dropna(subset=['DEP_DEL15'], inplace=True)

label = "DEP_DEL15"
df[label] = df[label].astype(int)

# Preprocess
df = pipeline.preprocessing(df)

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df[label].value_counts()}")

# --- Augmentation: inject nulls into selected features ---
# These features are realistic to be partially missing (e.g. tail number not always
# recorded, flight number changes, distance not always in booking systems)
rng = np.random.RandomState(SEED)
augment_features = [
    'TAIL_NUM',
    'DISTANCE',
    'OP_CARRIER_FL_NUM',
    'DAY_OF_MONTH',
]

# Inject 20% nulls per feature, independently
null_pct = 0.20
n_null = int(null_pct * len(df))
for feat in augment_features:
    null_idx = rng.choice(df.index, size=n_null, replace=False)
    df.loc[null_idx, feat] = np.nan

print(f"\nAfter augmentation (injected {null_pct:.0%} nulls each):")
for feat in augment_features:
    print(f"  {feat}: {df[feat].isna().mean():.2%} null")

print(f"N_TRIALS per model: {N_TRIALS}")

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
