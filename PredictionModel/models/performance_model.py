# Performance prediction model
# Predicts final training performance from early data (first 500k steps)

import numpy as np
from .base_predictor import BasePredictor


class PerformanceModel(BasePredictor):
    """Predicts final performance based on early training metrics."""

    def __init__(self, early_cutoff=500000):
        super().__init__()
        self.early_cutoff = early_cutoff
        self.feature_names = []

    def extract_features(self, df, target_col='reward_metric'):
        """
        Extracts features from early training data.
        Returns feature array or None if not enough data.
        """
        early = df[df['step'] <= self.early_cutoff]
        if len(early) < 5:
            return None

        values = early[target_col].values
        steps = early['step'].values

        features = []

        # Basic stats
        features.append(np.mean(values))
        features.append(np.std(values))
        features.append(np.min(values))
        features.append(np.max(values))
        features.append(np.median(values))

        # Trend (slope)
        if len(values) >= 2:
            steps_norm = (steps - steps[0]) / (steps[-1] - steps[0] + 1e-8)
            slope = np.polyfit(steps_norm, values, 1)[0]
            features.append(slope)
        else:
            features.append(0)

        # Improvement (late mean - early mean)
        mid = len(values) // 2
        if mid > 0:
            early_mean = np.mean(values[:mid])
            late_mean = np.mean(values[mid:])
            features.append(late_mean - early_mean)
        else:
            features.append(0)

        # Stability (std of last portion)
        tail = max(3, len(values) // 5)
        features.append(np.std(values[-tail:]))

        # Sample count and max step
        features.append(len(early))
        features.append(steps[-1])

        # Hyperparameters if available
        hp_cols = ['learning_rate', 'batch_size', 'buffer_size', 'num_epoch']
        for col in hp_cols:
            if col in early.columns:
                features.append(early[col].iloc[0])

        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median', 'slope', 'improvement',
            'tail_std', 'n_samples', 'max_step'
        ] + [c for c in hp_cols if c in early.columns]

        return np.array(features, dtype=np.float64)

    def compute_target(self, df, target_col='reward_metric'):
        """Computes final performance (mean of last 20% of training)."""
        max_step = df['step'].max()
        final = df[df['step'] >= max_step * 0.8]
        if len(final) == 0:
            return None
        return final[target_col].mean()

    def prepare_data(self, df):
        """
        Prepares training data from a dataframe with multiple runs.
        Returns (X, y, run_ids) or (None, None, None) if failed.
        """
        if 'run_id' not in df.columns:
            df = df.copy()
            df['run_id'] = 'single_run'

        X_list = []
        y_list = []
        run_ids = []

        for run_id in df['run_id'].unique():
            run_df = df[df['run_id'] == run_id].sort_values('step')

            # Skip if run is too short
            if run_df['step'].max() <= self.early_cutoff:
                continue

            features = self.extract_features(run_df)
            target = self.compute_target(run_df)

            if features is None or target is None:
                continue

            X_list.append(features)
            y_list.append(target)
            run_ids.append(run_id)

        if len(X_list) == 0:
            return None, None, None

        return np.vstack(X_list), np.array(y_list), run_ids

    def train(self, X, y):
        """Trains the model. Returns R-squared score."""
        if X is None or len(X) == 0:
            return None

        X_norm = self.normalize(X, fit=True)
        self.fit_ensemble(X_norm, y)

        # Cross-validate
        _, r2 = self.cross_validate(X_norm, y)
        return r2

    def predict(self, df):
        """Predicts final performance from early data."""
        if not self.is_trained:
            return None

        features = self.extract_features(df)
        if features is None:
            return None

        X = features.reshape(1, -1)
        X_norm = self.normalize(X, fit=False)
        return self.predict_ensemble(X_norm)[0]
