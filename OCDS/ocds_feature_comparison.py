"""
OCDS Feature Comparison Class

This class runs OCDS twice:
1. With all features except extended features
2. With all features (including extended features)

For each run, it prints the AUC score.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from OCDS.OCDS import OCDSClassifier


class OCDSFeatureComparison:
    """
    Compare OCDS performance with and without extended features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The complete dataframe with all features and target
    features_extended : list
        List of extended feature names to be excluded in the first run
    target_col : str
        Name of the target column
    n_estimators : int, default=10
        Number of estimators for OCDS
    max_depth : int, default=6
        Maximum depth for OCDS trees
    window_size : int, default=100
        Window size for drift detection
    drift_threshold : float, default=0.05
        Threshold for drift detection
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, 
                 df, 
                 features_extended, 
                 target_col='DEP_DEL15',
                 n_estimators=10,
                 max_depth=6,
                 window_size=100,
                 drift_threshold=0.05,
                 random_state=42):
        
        self.df = df.copy()
        self.features_extended = features_extended
        self.target_col = target_col
        
        # OCDS parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.random_state = random_state
        
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
        Train OCDS model and evaluate AUC.
        
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
        
        # Initialize OCDS
        model = OCDSClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            window_size=self.window_size,
            drift_threshold=self.drift_threshold,
            random_state=self.random_state
        )
        
        # Train using partial_fit (streaming fashion)
        model.partial_fit(X, y, classes=np.unique(y))
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X)
        
        # Calculate AUC using probabilities of positive class
        try:
            # For binary classification, use probability of class 1
            if y_pred_proba.shape[1] == 2:
                auc = roc_auc_score(y, y_pred_proba[:, 1])
            else:
                # For multi-class, use predict instead
                y_pred = model.predict(X)
                auc = roc_auc_score(y, y_pred)
        except ValueError as e:
            # Handle case where predictions are all the same class
            print(f"Warning: Could not calculate AUC for {run_name}: {e}")
            auc = 0.5
        
        return auc
    
    def run(self):
        """
        Run OCDS twice and print AUC scores.
        
        Returns
        -------
        dict
            Dictionary with AUC scores for both runs
        """
        print("=" * 70)
        print("OCDS Feature Comparison")
        print("=" * 70)
        
        # Run 1: Without extended features
        print("\n[1] Running OCDS WITHOUT extended features...")
        features_without_extended = self._get_features_without_extended()
        print(f"    Features used: {len(features_without_extended)}")
        print(f"    Excluded features: {self.features_extended}")
        
        self.auc_without_extended = self._train_and_evaluate(
            features_without_extended, 
            "without extended features"
        )
        print(f"    AUC (without extended): {self.auc_without_extended:.4f}")
        
        # Run 2: With all features
        print("\n[2] Running OCDS WITH all features (including extended)...")
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
