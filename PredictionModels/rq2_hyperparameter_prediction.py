import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.hyperparameter_data_loader import RQ2DataLoader


class RewardPredictor:

    FEATURE_NAMES = [
        'learning_rate', 'log_learning_rate', 'batch_size', 'log_batch_size',
        'buffer_size', 'log_buffer_size', 'num_epoch', 'lambd', 'gamma',
        'total_steps', 'log_total_steps', 'batch_buffer_ratio', 'effective_lr',
        'algo_PPO', 'algo_SAC', 'algo_POCA'
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.gb = None
        self.ridge = None

    def extract_features(self, runs):
        X_data = []
        y = []
        run_ids = []

        for run in runs:
            features = self._extract_run_features(run)
            X_data.append([features[name] for name in self.FEATURE_NAMES])
            y.append(run['target_reward'])
            run_ids.append(run['run_id'])

        X = np.array(X_data)
        y = np.array(y)

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        return X, y, run_ids

    def _extract_run_features(self, run):
        lr = run.get('learning_rate', 0) or 1e-6
        batch = run.get('batch_size', 0) or 1
        buffer = run.get('buffer_size', 0) or 1
        steps = run.get('total_steps', 0) or 1
        algo = run.get('algorithm', 'PPO').upper()

        return {
            'learning_rate': lr,
            'log_learning_rate': np.log10(max(lr, 1e-8)),
            'batch_size': batch,
            'log_batch_size': np.log2(max(batch, 1)),
            'buffer_size': buffer,
            'log_buffer_size': np.log2(max(buffer, 1)),
            'num_epoch': run.get('num_epoch', 0) or 0,
            'lambd': run.get('lambd', 0.95) or 0.95,
            'gamma': run.get('gamma', 0.99) or 0.99,
            'total_steps': steps,
            'log_total_steps': np.log10(max(steps, 1)),
            'batch_buffer_ratio': batch / max(buffer, 1),
            'effective_lr': lr * batch,
            'algo_PPO': 1 if algo == 'PPO' else 0,
            'algo_SAC': 1 if algo == 'SAC' else 0,
            'algo_POCA': 1 if algo == 'POCA' else 0,
        }

    def fit(self, X, y):
        X_norm = self.scaler.fit_transform(X)

        self.gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42
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

        mask = np.abs(y_true) > 0.01
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

        relative_errors = np.abs((y_true - y_pred) / (np.abs(y_true) + 0.01)) * 100
        within_25 = np.mean(relative_errors <= 25) * 100
        within_50 = np.mean(relative_errors <= 50) * 100

        return {
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'within_25': within_25,
            'within_50': within_50,
            'n': len(y_true)
        }


class RQ2Evaluator:

    def __init__(self, training_data_dir, shared_data_dir):
        self.loader = RQ2DataLoader(training_data_dir, shared_data_dir)

    def run(self):
        data = self.loader.load()
        results = {}

        for game_key, game_name in [('soccertwos', 'SoccerTwos'), ('worm', 'Worm')]:
            runs = data[game_key]
            if len(runs) < 5:
                continue

            metrics = self._evaluate_game(runs)
            if metrics:
                results[game_name] = {'n_runs': len(runs), 'metrics': metrics}

        self._print_results(results)

    def _evaluate_game(self, runs, n_folds=5):
        predictor = RewardPredictor()
        X, y, run_ids = predictor.extract_features(runs)

        if len(X) < 5:
            return None

        n_folds = min(n_folds, len(X) // 2)
        if n_folds < 2:
            return None

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            model = RewardPredictor()
            model.scaler.fit(X[train_idx])
            X_train_norm = model.scaler.transform(X[train_idx])
            X_test_norm = model.scaler.transform(X[test_idx])

            model.gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42
            )
            model.gb.fit(X_train_norm, y[train_idx])

            model.ridge = Ridge(alpha=1.0)
            model.ridge.fit(X_train_norm, y[train_idx])

            predictions[test_idx] = 0.7 * model.gb.predict(X_test_norm) + 0.3 * model.ridge.predict(X_test_norm)

        return MetricsCalculator.compute(y, predictions)

    def _print_results(self, results):
        print("\nRQ2: Reward Prediction Results\n")
        print(f"{'Game':<15} {'Runs':>8} {'R2':>10} {'MAE':>12} {'<25%':>10} {'<50%':>10}")
        for game, data in results.items():
            m = data['metrics']
            print(f"{game:<15} {data['n_runs']:>8} {m['r2']:>10.2f} {m['mae']:>12.4f} {m['within_25']:>9.1f}% {m['within_50']:>9.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', default='../training-data')
    parser.add_argument('--shared_data', default='../shared-data')
    args = parser.parse_args()

    evaluator = RQ2Evaluator(args.training_data, args.shared_data)
    evaluator.run()


if __name__ == "__main__":
    main()
