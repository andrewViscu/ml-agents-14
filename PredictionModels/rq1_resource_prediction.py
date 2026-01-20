import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.step_level_resource_loader import StepLevelResourceLoader, FeatureExtractor


class ResourcePredictor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.gb = None
        self.ridge = None

    def fit(self, X, y):
        X_norm = self.scaler.fit_transform(X)

        self.gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42
        )
        self.gb.fit(X_norm, y)

        self.ridge = Ridge(alpha=1.0)
        self.ridge.fit(X_norm, y)

    def predict(self, X):
        X_norm = self.scaler.transform(X)
        return 0.7 * self.gb.predict(X_norm) + 0.3 * self.ridge.predict(X_norm)


class MetricsCalculator:

    @staticmethod
    def compute(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        errors = np.abs(y_true - y_pred)
        mae = np.mean(errors)

        mask = np.abs(y_true) > 0.1
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

        relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + 0.1)) * 100
        within_10 = np.mean(relative_errors <= 10) * 100
        within_25 = np.mean(relative_errors <= 25) * 100

        return {
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'within_10': within_10,
            'within_25': within_25,
            'n': len(y_true)
        }


class RQ1Evaluator:

    TARGETS = [
        ('memory_usage_avg_mb', 'Memory (MB)'),
        ('cpu_usage_avg_percent', 'CPU (%)'),
        ('gpu_usage_avg_mb', 'GPU (MB)'),
    ]

    def __init__(self, training_data_dir):
        self.loader = StepLevelResourceLoader(training_data_dir)
        self.extractor = FeatureExtractor()

    def run(self):
        df = self.loader.load()
        if len(df) == 0:
            print("No data loaded")
            return

        results = {}
        for game in ['SoccerTwos', 'Worm']:
            game_df = df[df['game'] == game].copy()
            n_runs = game_df['run_id'].nunique()
            results[game] = {'n_runs': n_runs, 'n_points': len(game_df), 'metrics': {}}

            for target_col, target_name in self.TARGETS:
                metrics = self._evaluate_target(game_df, target_col)
                if metrics:
                    results[game]['metrics'][target_name] = metrics

        self._print_results(results)

    def _evaluate_target(self, game_df, target_col):
        target_df = game_df.dropna(subset=[target_col])
        if len(target_df) < 50:
            return None

        X, y, metadata = self.extractor.extract(target_df, target_col)
        if X is None:
            return None

        run_ids = metadata['run_ids']
        unique_runs = list(set(run_ids))
        run_to_group = {r: i for i, r in enumerate(unique_runs)}
        groups = np.array([run_to_group[r] for r in run_ids])

        n_folds = min(5, len(unique_runs))
        if n_folds < 2:
            return None

        cv = GroupKFold(n_splits=n_folds)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X, y, groups):
            model = ResourcePredictor()
            model.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(X[test_idx])

        return MetricsCalculator.compute(y, predictions)

    def _print_results(self, results):
        print("\nRQ1: Resource Prediction Results\n")
        for game, data in results.items():
            print(f"{game} ({data['n_runs']} runs, {data['n_points']} data points)")
            print(f"{'Resource':<15} {'R2':>8} {'MAE':>12} {'<10%':>10} {'<25%':>10}")
            for target_name, m in data['metrics'].items():
                print(f"{target_name:<15} {m['r2']:>8.2f} {m['mae']:>12.2f} {m['within_10']:>9.1f}% {m['within_25']:>9.1f}%")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', default='../training-data')
    args = parser.parse_args()

    evaluator = RQ1Evaluator(args.training_data)
    evaluator.run()


if __name__ == "__main__":
    main()
