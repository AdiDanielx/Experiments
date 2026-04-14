"""
EML: Ensemble of Meta-Learners
Based on: https://arxiv.org/abs/2006.15334

This implementation provides an ensemble approach that combines multiple
base learners with a meta-learner for improved predictions.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict


class EMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Meta-Learners Classifier.
    
    Uses multiple diverse base learners and combines their predictions
    using a meta-learner (stacking approach).
    
    Parameters
    ----------
    base_learners : list of estimators, default=None
        List of base learners. If None, uses default set of diverse learners
    meta_learner : estimator, default=None
        Meta-learner to combine base predictions. If None, uses LogisticRegression
    cv : int, default=5
        Number of cross-validation folds for generating meta-features
    use_proba : bool, default=True
        Whether to use predicted probabilities as meta-features
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self,
                 base_learners=None,
                 meta_learner=None,
                 cv=5,
                 use_proba=True,
                 random_state=None):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.cv = cv
        self.use_proba = use_proba
        self.random_state = random_state
        
        self.base_learners_ = None
        self.meta_learner_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def _get_default_base_learners(self):
        """Get default set of diverse base learners."""
        return [
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                random_state=self.random_state
            ),
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state + 1 if self.random_state else None
            ),
            RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state + 2 if self.random_state else None
            ),
            RandomForestClassifier(
                n_estimators=30,
                max_depth=10,
                random_state=self.random_state + 3 if self.random_state else None
            ),
        ]
    
    def _get_default_meta_learner(self):
        """Get default meta-learner."""
        return LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )
    
    def _generate_meta_features(self, X, y, fit_base=True):
        """
        Generate meta-features from base learners.
        
        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target values
        fit_base : bool
            Whether to fit base learners
        
        Returns
        -------
        meta_features : ndarray
            Meta-features for training meta-learner
        """
        n_samples = X.shape[0]
        
        if fit_base:
            # Use cross-validation to generate out-of-fold predictions
            meta_features_list = []
            
            for i, learner in enumerate(self.base_learners_):
                if self.use_proba:
                    # Get probability predictions
                    preds = cross_val_predict(
                        learner, X, y,
                        cv=self.cv,
                        method='predict_proba'
                    )
                    meta_features_list.append(preds)
                else:
                    # Get class predictions
                    preds = cross_val_predict(
                        learner, X, y,
                        cv=self.cv,
                        method='predict'
                    )
                    # One-hot encode
                    preds_onehot = np.zeros((n_samples, self.n_classes_))
                    for j, cls in enumerate(self.classes_):
                        preds_onehot[preds == cls, j] = 1
                    meta_features_list.append(preds_onehot)
                
                # Fit on full data
                learner.fit(X, y)
            
            # Concatenate all meta-features
            meta_features = np.hstack(meta_features_list)
        else:
            # Use fitted base learners to generate predictions
            meta_features_list = []
            
            for learner in self.base_learners_:
                if self.use_proba:
                    preds = learner.predict_proba(X)
                else:
                    preds = learner.predict(X)
                    # One-hot encode
                    preds_onehot = np.zeros((n_samples, self.n_classes_))
                    for j, cls in enumerate(self.classes_):
                        preds_onehot[preds == cls, j] = 1
                    preds = preds_onehot
                
                meta_features_list.append(preds)
            
            meta_features = np.hstack(meta_features_list)
        
        return meta_features
    
    def fit(self, X, y):
        """
        Fit the EML model.
        
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
        
        # Initialize base learners
        if self.base_learners is None:
            self.base_learners_ = self._get_default_base_learners()
        else:
            self.base_learners_ = [clone(learner) for learner in self.base_learners]
        
        # Initialize meta-learner
        if self.meta_learner is None:
            self.meta_learner_ = self._get_default_meta_learner()
        else:
            self.meta_learner_ = clone(self.meta_learner)
        
        # Generate meta-features and fit base learners
        meta_features = self._generate_meta_features(X, y, fit_base=True)
        
        # Fit meta-learner
        self.meta_learner_.fit(meta_features, y)
        
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
            
            # Initialize learners
            if self.base_learners is None:
                self.base_learners_ = self._get_default_base_learners()
            else:
                self.base_learners_ = [clone(learner) for learner in self.base_learners]
            
            if self.meta_learner is None:
                self.meta_learner_ = self._get_default_meta_learner()
            else:
                self.meta_learner_ = clone(self.meta_learner)
            
            self.is_fitted_ = True
        
        # Fit base learners on new data
        for learner in self.base_learners_:
            learner.fit(X, y)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, y, fit_base=False)
        
        # Update meta-learner
        # Note: LogisticRegression doesn't have partial_fit, so we refit
        # For production, consider using SGDClassifier as meta-learner
        self.meta_learner_.fit(meta_features, y)
        
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
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, None, fit_base=False)
        
        # Predict with meta-learner
        if hasattr(self.meta_learner_, 'predict_proba'):
            return self.meta_learner_.predict_proba(meta_features)
        else:
            # If meta-learner doesn't support predict_proba, use predict
            preds = self.meta_learner_.predict(meta_features)
            proba = np.zeros((len(X), self.n_classes_))
            for i, cls in enumerate(self.classes_):
                proba[preds == cls, i] = 1.0
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
