#!/usr/bin/env python3
"""Run all valid combinations on MovieLens dataset."""
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
print("Downloading MovieLens 20M dataset...")
path = kagglehub.dataset_download("grouplens/movielens-20m-dataset")
print(f"Dataset path: {path}")

ratings = pd.read_csv(os.path.join(path, "rating.csv"))
tags = pd.read_csv(os.path.join(path, "tag.csv"))
print(f"Ratings: {ratings.shape}, Tags: {tags.shape}")

# Convert datetime strings to unix timestamps (20M dataset uses datetime format)
print("Converting timestamps...")
ratings['timestamp'] = pd.to_datetime(ratings['timestamp']).astype(np.int64) // 10**9
tags['timestamp'] = pd.to_datetime(tags['timestamp']).astype(np.int64) // 10**9

# --- Feature engineering (same as notebook) ---
print("Engineering features...")
cutoff_date = ratings['timestamp'].quantile(0.8)
target_date = ratings['timestamp'].quantile(0.9)

# Base rating features
user_stats = ratings[ratings['timestamp'] < cutoff_date].groupby('userId').agg({
    'rating': ['count', 'mean', 'std'],
    'timestamp': ['min', 'max']
}).round(3)

user_stats.columns = ['rating_count', 'rating_mean', 'rating_std',
                       'first_rating', 'last_rating']
df = user_stats.reset_index()
df['days_active'] = (df['last_rating'] - df['first_rating']) / (24 * 60 * 60)
df['rating_frequency'] = df['rating_count'] / df['days_active'].clip(lower=1)

# Target: future activity above median
future_activity = ratings[
    (ratings['timestamp'] >= cutoff_date) &
    (ratings['timestamp'] < target_date)
].groupby('userId')['rating'].count().reset_index()
future_activity.columns = ['userId', 'future_ratings']
future_activity['TARGET'] = (future_activity['future_ratings'] >
                              future_activity['future_ratings'].median()).astype(int)

df = df.merge(future_activity[['userId', 'TARGET']], on='userId', how='left')
df['TARGET'] = df['TARGET'].fillna(0)

# Tag-based features (extended)
tag_activity = tags[tags['timestamp'] < cutoff_date].groupby('userId').agg({
    'tag': ['count', 'nunique'],
    'timestamp': ['min', 'max']
})
tag_activity.columns = ['tag_count', 'unique_tags', 'first_tag', 'last_tag']
tag_activity = tag_activity.reset_index()

tag_activity['days_tagging'] = (tag_activity['last_tag'] - tag_activity['first_tag']) / (24 * 60 * 60)
tag_activity['tag_frequency'] = tag_activity['tag_count'] / tag_activity['days_tagging'].clip(lower=1)

tag_lengths = tags[tags['timestamp'] < cutoff_date].groupby('userId')['tag'].apply(
    lambda x: np.mean([len(str(t)) for t in x])
).reset_index()
tag_lengths.columns = ['userId', 'avg_tag_length']

tag_activity = tag_activity.merge(tag_lengths, on='userId', how='left')
df = df.merge(tag_activity, on='userId', how='left')

# Keep only the columns used in the notebook
columns_to_keep = [
    'rating_count', 'rating_mean', 'rating_std',
    'days_active', 'rating_frequency',
    'tag_count', 'unique_tags', 'avg_tag_length',
    'tag_frequency', 'last_tag', 'TARGET'
]
df = df[columns_to_keep]
df = pipeline.preprocessing(df)

label = "TARGET"
df[label] = df[label].astype(int)

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
