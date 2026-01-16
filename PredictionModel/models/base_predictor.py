# Base class for prediction models

from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class BasePredictor(ABC):
    """Base class with shared ML functionality."""

    def __init__(self):
        self.model = None
        self.ridge = None
        self.feature_mean = None
        self.feature_std = None
        self.is_trained = False

    def normalize(self, X, fit=False):
        """Z-score normalization."""
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
        return (X - self.feature_mean) / self.feature_std

    def create_models(self):
        """Creates Ridge and Gradient Boosting models."""
        ridge = Ridge(alpha=10.0)
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        return ridge, gb

    def fit_ensemble(self, X, y):
        """Fits both models on the data."""
        self.ridge, self.model = self.create_models()
        self.ridge.fit(X, y)
        self.model.fit(X, y)
        self.is_trained = True

    def predict_ensemble(self, X):
        """Weighted prediction from both models."""
        if not self.is_trained:
            return None
        pred_ridge = self.ridge.predict(X)
        pred_gb = self.model.predict(X)
        return 0.3 * pred_ridge + 0.7 * pred_gb

    def cross_validate(self, X, y, n_folds=5):
        """K-fold cross-validation. Returns (predictions, r2_score)."""
        n = len(X)
        if n < n_folds:
            n_folds = n

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        predictions = np.zeros(n)

        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            ridge, gb = self.create_models()
            ridge.fit(X_train, y_train)
            gb.fit(X_train, y_train)

            predictions[test_idx] = 0.3 * ridge.predict(X_test) + 0.7 * gb.predict(X_test)

        # Compute R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return predictions, r2

    def get_feature_importance(self):
        """Returns feature importances from gradient boosting model."""
        if self.model is None:
            return None
        return self.model.feature_importances_

    @abstractmethod
    def train(self, X, y):
        """Train the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions. Must be implemented by subclasses."""
        pass
