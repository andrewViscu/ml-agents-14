# Resource prediction model
# Predicts training time, CPU, RAM, GPU usage from configuration

import numpy as np
from .base_predictor import BasePredictor


class ResourceModel(BasePredictor):
    """Predicts resource requirements from training configuration."""

    def __init__(self):
        super().__init__()
        self.models = {}  # One model per target
        self.target_names = []
        self.feature_names = []

    def train(self, X, Y, feature_names, target_names):
        """
        Trains a model for each resource target.
        X: feature matrix (n_samples, n_features)
        Y: target matrix (n_samples, n_targets)
        Returns dict with R2 scores for each target.
        """
        if X is None or len(X) == 0:
            return None

        self.feature_names = feature_names
        self.target_names = target_names

        # Normalize features
        X_norm = self.normalize(X, fit=True)

        results = {}

        for i, target in enumerate(target_names):
            y = Y[:, i]

            # Skip if no variance
            if np.std(y) < 1e-8:
                continue

            # Apply log transform for skewed targets
            use_log = 'time' in target.lower() or 'ram' in target.lower() or 'mb' in target.lower()
            if use_log:
                y_train = np.log1p(np.maximum(0, y))
            else:
                y_train = y

            # Normalize target
            y_mean = np.mean(y_train)
            y_std = np.std(y_train) + 1e-8
            y_norm = (y_train - y_mean) / y_std

            # Train ensemble
            ridge, gb = self.create_models()
            ridge.fit(X_norm, y_norm)
            gb.fit(X_norm, y_norm)

            self.models[target] = {
                'ridge': ridge,
                'gb': gb,
                'y_mean': y_mean,
                'y_std': y_std,
                'use_log': use_log
            }

            # Cross-validate
            _, r2 = self.cross_validate(X_norm, y_norm)
            results[target] = r2

        self.is_trained = True
        return results

    def predict(self, X):
        """Predicts resource usage for given features."""
        if not self.is_trained:
            return None

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_norm = self.normalize(X, fit=False)

        predictions = {}
        for target, model_info in self.models.items():
            # Ensemble prediction
            pred_ridge = model_info['ridge'].predict(X_norm)
            pred_gb = model_info['gb'].predict(X_norm)
            pred_norm = 0.3 * pred_ridge + 0.7 * pred_gb

            # Reverse normalization
            pred = pred_norm * model_info['y_std'] + model_info['y_mean']

            # Reverse log transform
            if model_info['use_log']:
                pred = np.expm1(np.clip(pred, -20, 20))

            predictions[target] = max(0, float(pred[0]))

        return predictions

    def get_feature_importance(self, target=None):
        """Returns feature importances for a target."""
        if not self.is_trained:
            return None

        if target is None:
            target = list(self.models.keys())[0]

        if target not in self.models:
            return None

        importances = self.models[target]['gb'].feature_importances_
        return dict(zip(self.feature_names, importances))
