#!/usr/bin/env python3
"""Run all valid combinations on HR Analytics dataset."""
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
path = kagglehub.dataset_download("arashnic/hr-analytics-job-change-of-data-scientists")
csv_path = os.path.join(path, "aug_train.csv")
df = pd.read_csv(csv_path)

# Drop enrollee_id (unique identifier, not predictive)
df = df.drop(columns=['enrollee_id'])
df = pipeline.preprocessing(df)

label = "target"
# Ensure target is int
df[label] = df[label].astype(int)

print(f"Dataset shape: {df.shape}")
print(f"N_TRIALS per model: {N_TRIALS}")
print(f"Target distribution:\n{df[label].value_counts()}")

# --- EML Feature Comparison ---
print(f"\n{'='*120}")
print("EML FEATURE COMPARISON")
print(f"{'='*120}")

from EML.eml_feature_comparison import EMLFeatureComparison

# Define extended features (fill in the list with feature names)
extended_features = []  # TODO: Add extended feature names here

print(f"Extended features: {extended_features}")
print(f"Running EML comparison...")

# Initialize and run EML comparison
eml_comparison = EMLFeatureComparison(
    df=df.copy(),
    features_extended=extended_features,
    target_col=label,
    base_learners=None,
    meta_learner=None,
    cv=5,
    use_proba=True,
    
    
    
)

# Run the comparison
result = eml_comparison.run()

print(f"\n{'='*120}")
print("EML SUMMARY")
print(f"{'='*120}")
print(f"AUC (Basic features only):  {result['auc_without_extended']:.4f}")
print(f"AUC (All features):         {result['auc_with_all']:.4f}")
print(f"Difference:                 {result['difference']:+.4f}")
print(f"{'='*120}")

print("\nEML Analysis Complete!")
