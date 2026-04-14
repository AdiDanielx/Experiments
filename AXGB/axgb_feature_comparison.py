"""
AXGB Feature Comparison Class

This class runs Adaptive XGBoost twice:
1. With all features except extended features
2. With all features (including extended features)

For each run, it prints the AUC score.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from axgb.adaptive_xgboost import AdaptiveXGBoostClassifier


class AXGBFeatureComparison:
    """
    Compare AXGB performance with and without extended features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The complete dataframe with all features and target
    features_extended : list
        List of extended feature names to be excluded in the first run
    target_col : str
        Name of the target column
    n_estimators : int, default=30
        Number of estimators for AXGB
    learning_rate : float, default=0.3
        Learning rate for AXGB
    max_depth : int, default=6
        Maximum depth for AXGB trees
    max_window_size : int, default=1000
        Maximum window size for AXGB
    min_window_size : int, default=None
        Minimum window size for AXGB
    detect_drift : bool, default=False
        Whether to detect drift in AXGB
    update_strategy : str, default='replace'
        Update strategy for AXGB ('push' or 'replace')
    """
    
    def __init__(self, 
                 df, 
                 features_extended, 
                 target_col='DEP_DEL15',
                 n_estimators=30,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 detect_drift=False,
                 update_strategy='replace'):
        
        self.df = df.copy()
        self.features_extended = features_extended
        self.target_col = target_col
        
        # AXGB parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.detect_drift = detect_drift
        self.update_strategy = update_strategy
        
        # Results storage
        self.auc_without_extended = None
        self.auc_with_all = None
        
    def _get_features_without_extended(self):
        """Get list of features excluding extended features and target."""
        all_features = [col for col in self.df.columns if col != self.target_col]
        features_without_extended = [f for f in all_features if f not in self.features_extended]
        return features_without_extended
    
    def _get_all_features(self):
        """Get list of all features excluding target."""
        return [col for col in self.df.columns if col != self.target_col]
    
    def _train_and_evaluate(self, features, run_name):
        """
        Train AXGB model and evaluate AUC.
        
        Parameters
        ----------
        features : list
            List of feature names to use
        run_name : str
            Name of the run for printing
            
        Returns
        -------
        float
            AUC score
        """
        # Prepare data
        X = self.df[features].values
        y = self.df[self.target_col].values
        
        # Initialize AXGB
        model = AdaptiveXGBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_window_size=self.max_window_size,
            min_window_size=self.min_window_size,
            detect_drift=self.detect_drift,
            update_strategy=self.update_strategy
        )
        
        # Train using partial_fit (streaming fashion)
        model.partial_fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y, y_pred)
        except ValueError as e:
            # Handle case where predictions are all the same class
            print(f"Warning: Could not calculate AUC for {run_name}: {e}")
            auc = 0.5
        
        return auc
    
    def run(self):
        """
        Run AXGB twice and print AUC scores.
        
        Returns
        -------
        dict
            Dictionary with AUC scores for both runs
        """
        print("=" * 70)
        print("AXGB Feature Comparison")
        print("=" * 70)
        
        # Run 1: Without extended features
        print("\n[1] Running AXGB WITHOUT extended features...")
        features_without_extended = self._get_features_without_extended()
        print(f"    Features used: {len(features_without_extended)}")
        print(f"    Excluded features: {self.features_extended}")
        
        self.auc_without_extended = self._train_and_evaluate(
            features_without_extended, 
            "without extended features"
        )
        print(f"    AUC (without extended): {self.auc_without_extended:.4f}")
        
        # Run 2: With all features
        print("\n[2] Running AXGB WITH all features (including extended)...")
        all_features = self._get_all_features()
        print(f"    Features used: {len(all_features)}")
        
        self.auc_with_all = self._train_and_evaluate(
            all_features, 
            "with all features"
        )
        print(f"    AUC (with all features): {self.auc_with_all:.4f}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"AUC without extended features: {self.auc_without_extended:.4f}")
        print(f"AUC with all features:          {self.auc_with_all:.4f}")
        print(f"Difference:                     {self.auc_with_all - self.auc_without_extended:+.4f}")
        print("=" * 70)
        
        return {
            'auc_without_extended': self.auc_without_extended,
            'auc_with_all': self.auc_with_all,
            'difference': self.auc_with_all - self.auc_without_extended
        }
