"""
GBDT-IL: Gradient Boosting Decision Tree with Incremental Learning
Based on: https://www.mdpi.com/1424-8220/24/7/2083

This implementation provides an incremental learning version of GBDT that can
update the model with new data without full retraining.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import deque


class GBDTILClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Boosting Decision Tree with Incremental Learning.
    
    Maintains a sliding window of recent trees and incrementally updates
    the ensemble as new data arrives.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting iterations
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree
    max_depth : int, default=3
        Maximum depth of the individual regression estimators
    subsample : float, default=1.0
        Fraction of samples to be used for fitting the individual base learners
    max_window_size : int, default=50
        Maximum number of trees to keep in the ensemble
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 subsample=1.0,
                 max_window_size=50,
                 min_samples_split=2,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_window_size = max_window_size
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.estimators_ = deque(maxlen=max_window_size)
        self.init_estimator_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def _init_decision_function(self, y):
        """Initialize the decision function with class priors."""
        if self.n_classes_ == 2:
            # Binary classification
            pos_ratio = np.mean(y == self.classes_[1])
            # Log odds
            init_score = np.log(pos_ratio / (1 - pos_ratio + 1e-10))
            return np.full(len(y), init_score)
        else:
            # Multi-class: one-vs-rest
            init_scores = np.zeros((len(y), self.n_classes_))
            for k in range(self.n_classes_):
                pos_ratio = np.mean(y == self.classes_[k])
                init_scores[:, k] = np.log(pos_ratio + 1e-10)
            return init_scores
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _softmax(self, x):
        """Numerically stable softmax function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _compute_gradients(self, y, raw_predictions):
        """Compute gradients for the loss function."""
        if self.n_classes_ == 2:
            # Binary cross-entropy gradients
            p = self._sigmoid(raw_predictions)
            y_binary = (y == self.classes_[1]).astype(float)
            gradients = y_binary - p
            return gradients
        else:
            # Multi-class cross-entropy gradients
            p = self._softmax(raw_predictions)
            gradients = np.zeros_like(raw_predictions)
            for k in range(self.n_classes_):
                y_k = (y == self.classes_[k]).astype(float)
                gradients[:, k] = y_k - p[:, k]
            return gradients
    
    def fit(self, X, y):
        """
        Fit the GBDT-IL model.
        
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
        self.n_classes_ = len(self.classes_)
        
        # Initialize decision function
        raw_predictions = self._init_decision_function(y)
        
        # Build initial ensemble
        self.estimators_.clear()
        
        rng = np.random.RandomState(self.random_state)
        
        for i in range(min(self.n_estimators, self.max_window_size)):
            # Compute gradients
            gradients = self._compute_gradients(y, raw_predictions)
            
            # Subsample
            if self.subsample < 1.0:
                n_samples = int(self.subsample * len(X))
                indices = rng.choice(len(X), size=n_samples, replace=False)
                X_sub = X[indices]
                grad_sub = gradients[indices] if gradients.ndim == 1 else gradients[indices]
            else:
                X_sub = X
                grad_sub = gradients
            
            # Fit tree to gradients
            if self.n_classes_ == 2:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state + i if self.random_state else None
                )
                tree.fit(X_sub, grad_sub)
                self.estimators_.append(tree)
                
                # Update predictions
                raw_predictions += self.learning_rate * tree.predict(X)
            else:
                # Multi-class: fit one tree per class
                trees = []
                for k in range(self.n_classes_):
                    tree = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        random_state=self.random_state + i * self.n_classes_ + k if self.random_state else None
                    )
                    tree.fit(X_sub, grad_sub[:, k])
                    trees.append(tree)
                    
                    # Update predictions
                    raw_predictions[:, k] += self.learning_rate * tree.predict(X)
                
                self.estimators_.append(trees)
        
        self.is_fitted_ = True
        return self
    
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
            self.n_classes_ = len(self.classes_)
            self.is_fitted_ = True
        
        # Get current predictions
        if len(self.estimators_) == 0:
            raw_predictions = self._init_decision_function(y)
        else:
            raw_predictions = self._raw_predict(X)
        
        # Compute gradients
        gradients = self._compute_gradients(y, raw_predictions)
        
        # Fit new tree(s) to gradients
        rng = np.random.RandomState(self.random_state)
        
        if self.n_classes_ == 2:
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X, gradients)
            self.estimators_.append(tree)
        else:
            trees = []
            for k in range(self.n_classes_):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_state=rng.randint(0, 10000)
                )
                tree.fit(X, gradients[:, k])
                trees.append(tree)
            self.estimators_.append(trees)
        
        return self
    
    def _raw_predict(self, X):
        """Compute raw predictions (before applying link function)."""
        X = np.asarray(X)
        
        if len(self.estimators_) == 0:
            if self.n_classes_ == 2:
                return np.zeros(len(X))
            else:
                return np.zeros((len(X), self.n_classes_))
        
        if self.n_classes_ == 2:
            raw_predictions = np.zeros(len(X))
            for tree in self.estimators_:
                raw_predictions += self.learning_rate * tree.predict(X)
        else:
            raw_predictions = np.zeros((len(X), self.n_classes_))
            for trees in self.estimators_:
                for k, tree in enumerate(trees):
                    raw_predictions[:, k] += self.learning_rate * tree.predict(X)
        
        return raw_predictions
    
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
        
        raw_predictions = self._raw_predict(X)
        
        if self.n_classes_ == 2:
            proba_pos = self._sigmoid(raw_predictions)
            return np.vstack([1 - proba_pos, proba_pos]).T
        else:
            return self._softmax(raw_predictions)
    
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
