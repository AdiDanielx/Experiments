"""
OCDS: Online Concept Drift Detection and Streaming
Based on: https://www.ijcai.org/proceedings/2019/346

This implementation provides an online learning classifier with concept drift detection.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import deque


class OCDSClassifier(BaseEstimator, ClassifierMixin):
    """
    Online Concept Drift Detection and Streaming Classifier.
    
    Uses an ensemble of decision trees with drift detection based on error rate monitoring.
    When drift is detected, the model adapts by updating the ensemble.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of base estimators in the ensemble
    max_depth : int, default=6
        Maximum depth of decision trees
    window_size : int, default=100
        Size of the sliding window for drift detection
    drift_threshold : float, default=0.05
        Threshold for drift detection (change in error rate)
    min_samples_split : int, default=2
        Minimum samples required to split a node
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, 
                 n_estimators=10,
                 max_depth=6,
                 window_size=100,
                 drift_threshold=0.05,
                 min_samples_split=2,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.estimators_ = []
        self.weights_ = []
        self.error_window_ = deque(maxlen=window_size)
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        
    def _initialize_ensemble(self, X, y):
        """Initialize the ensemble with base estimators."""
        self.estimators_ = []
        self.weights_ = []
        
        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Bootstrap sample
            n_samples = len(X)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
            self.weights_.append(1.0)
        
        # Normalize weights
        self.weights_ = np.array(self.weights_) / np.sum(self.weights_)
    
    def _detect_drift(self):
        """Detect concept drift based on error rate changes."""
        if len(self.error_window_) < self.window_size:
            return False
        
        errors = np.array(self.error_window_)
        # Split window in half
        mid = len(errors) // 2
        old_error = np.mean(errors[:mid])
        new_error = np.mean(errors[mid:])
        
        # Drift detected if error rate increases significantly
        return (new_error - old_error) > self.drift_threshold
    
    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        classes : array-like, default=None
            Classes across all calls to partial_fit
        
        Returns
        -------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if not self.is_fitted_:
            self.n_features_in_ = X.shape[1]
            self.classes_ = classes if classes is not None else np.unique(y)
            self._initialize_ensemble(X, y)
            self.is_fitted_ = True
            return self
        
        # Make predictions and calculate error
        y_pred = self.predict(X)
        error = np.mean(y_pred != y)
        self.error_window_.append(error)
        
        # Detect drift
        if self._detect_drift():
            # Reinitialize ensemble on drift
            self._initialize_ensemble(X, y)
            self.error_window_.clear()
        else:
            # Update existing estimators
            for i, estimator in enumerate(self.estimators_):
                # Update with new data
                n_samples = len(X)
                indices = np.random.choice(n_samples, size=min(n_samples, 50), replace=True)
                X_update = X[indices]
                y_update = y[indices]
                
                # Retrain estimator
                estimator.fit(X_update, y_update)
        
        return self
    
    def fit(self, X, y):
        """
        Fit the model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._initialize_ensemble(X, y)
        self.is_fitted_ = True
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Weighted voting
        proba = np.zeros((n_samples, n_classes))
        
        for estimator, weight in zip(self.estimators_, self.weights_):
            est_proba = estimator.predict_proba(X)
            proba += weight * est_proba
        
        return proba
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
