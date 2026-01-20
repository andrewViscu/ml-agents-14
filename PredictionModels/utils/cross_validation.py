import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class EnsemblePredictor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=3, random_state=42
        )
        self.ridge = Ridge(alpha=1.0)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.gb.fit(X_scaled, y)
        self.ridge.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return 0.7 * self.gb.predict(X_scaled) + 0.3 * self.ridge.predict(X_scaled)


class CrossValidator:

    @staticmethod
    def kfold(X, y, n_folds=5):
        n_folds = min(n_folds, len(X) // 2)
        if n_folds < 2:
            return None

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            model = EnsemblePredictor()
            model.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(X[test_idx])

        return predictions

    @staticmethod
    def group_kfold(X, y, groups, n_folds=5):
        unique_groups = len(set(groups))
        n_folds = min(n_folds, unique_groups)
        if n_folds < 2:
            return None

        cv = GroupKFold(n_splits=n_folds)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X, y, groups):
            model = EnsemblePredictor()
            model.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(X[test_idx])

        return predictions


class Metrics:

    @staticmethod
    def compute(y_true, y_pred, thresholds=(10, 25)):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        mae = np.mean(np.abs(y_true - y_pred))

        rel_err = np.abs((y_true - y_pred) / (np.abs(y_true) + 0.1)) * 100
        within = {t: np.mean(rel_err <= t) * 100 for t in thresholds}

        return {'r2': r2, 'mae': mae, **{f'within_{t}': within[t] for t in thresholds}}
