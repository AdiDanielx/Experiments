#!/usr/bin/env python3
"""Run all valid combinations on DiabetesRecord dataset."""
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
MIN_GROUP_PCT = 0.07   # Min 7% of total population per test group

pipeline = GenericDataPipeline()

# --- Load data ---
path = kagglehub.dataset_download("brandao/diabetes")
csv_path = os.path.join(path, "diabetic_data.csv")
df = pd.read_csv(csv_path)

# Target: readmitted (NO=0, <30 or >30 = 1)
df['readmitted'] = df['readmitted'].replace({'NO': 0, '<30': 1, '>30': 1}).astype(int)

# Drop columns as in notebook
columns_to_drop = ['encounter_id', 'patient_nbr', 'number_inpatient', 'number_emergency', 'discharge_disposition_id']
df = df.drop(columns=columns_to_drop)
df = pipeline.preprocessing(df)
label = "readmitted"
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

