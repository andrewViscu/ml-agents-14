import argparse
import os
import sys
import numpy as np
import pandas as pd
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class TimeSeriesRewardLoader:

    def __init__(self, training_data_dir, shared_data_dir):
        self.training_data_dir = training_data_dir
        self.shared_data_dir = shared_data_dir

    def load(self, cutoff_steps):
        runs = []
        seen_ids = set()

        for run in self._load_from_single_run(cutoff_steps):
            if run['run_id'] not in seen_ids:
                runs.append(run)
                seen_ids.add(run['run_id'])

        for run in self._load_from_training_data(cutoff_steps):
            if run['run_id'] not in seen_ids:
                runs.append(run)
                seen_ids.add(run['run_id'])

        return runs

    def _load_from_single_run(self, cutoff_steps):
        runs = []
        timeseries_path = os.path.join(self.shared_data_dir, "single_run", "multiple_runs_with_steps.csv")

        if os.path.exists(timeseries_path):
            try:
                df = pd.read_csv(timeseries_path)
                step_col = self._find_column(df, ['Step', 'step', 'steps'])
                reward_col = self._find_column(df, ['cumulative_reward', 'mean_reward', 'reward'])
                run_col = self._find_column(df, ['RunID', 'run_id', 'run'])

                if step_col and reward_col and run_col:
                    for run_id, group in df.groupby(run_col):
                        run = self._process_group(group, run_id, step_col, reward_col, cutoff_steps)
                        if run:
                            runs.append(run)
            except Exception:
                pass

        return runs

    def _load_from_training_data(self, cutoff_steps):
        runs = []
        files = glob(os.path.join(self.training_data_dir, "*.csv"))
        files = [f for f in files if 'hardware' not in f.lower()]

        for filepath in files:
            run = self._process_timeseries_file(filepath, cutoff_steps)
            if run:
                runs.append(run)
        return runs

    def _process_timeseries_file(self, filepath, cutoff_steps):
        try:
            df = pd.read_csv(filepath)
            run_id = os.path.splitext(os.path.basename(filepath))[0]

            step_col = self._find_column(df, ['step', 'steps', 'training_step'])
            reward_col = self._find_column(df, ['mean_reward', 'reward', 'mean_group_reward'])

            if not step_col or not reward_col:
                return None

            return self._process_group(df, run_id, step_col, reward_col, cutoff_steps)
        except Exception:
            return None

    def _process_group(self, df, run_id, step_col, reward_col, cutoff_steps):
        df = df.sort_values(step_col)
        max_step = df[step_col].max()

        if max_step < cutoff_steps:
            return None

        early_df = df[df[step_col] <= cutoff_steps]
        final_df = df[df[step_col] >= max_step * 0.9]

        if len(early_df) < 3 or len(final_df) < 1:
            return None

        early_rewards = early_df[reward_col].dropna()
        final_rewards = final_df[reward_col].dropna()

        if len(early_rewards) < 3 or len(final_rewards) < 1:
            return None

        features = {
            'run_id': run_id,
            'early_mean': early_rewards.mean(),
            'early_std': early_rewards.std(),
            'early_min': early_rewards.min(),
            'early_max': early_rewards.max(),
            'early_last': early_rewards.iloc[-1],
            'early_trend': self._compute_trend(early_df[step_col].values, early_rewards.values),
            'n_early_points': len(early_rewards),
            'final_reward': final_rewards.mean(),
        }

        return features

    def _compute_trend(self, steps, rewards):
        if len(steps) < 2:
            return 0
        steps_norm = (steps - steps.min()) / (steps.max() - steps.min() + 1e-8)
        coef = np.polyfit(steps_norm, rewards, 1)
        return coef[0]

    def _find_column(self, df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None


class CutoffPredictor:

    FEATURE_NAMES = ['early_mean', 'early_std', 'early_min', 'early_max',
                     'early_last', 'early_trend', 'n_early_points']

    def extract_features(self, runs):
        X = np.array([[r[f] for f in self.FEATURE_NAMES] for r in runs])
        y = np.array([r['final_reward'] for r in runs])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        return X, y

    def evaluate(self, runs, n_folds=5):
        if len(runs) < 10:
            return None

        X, y = self.extract_features(runs)
        n_folds = min(n_folds, len(runs) // 2)

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                           learning_rate=0.1, random_state=42)
            gb.fit(X_train, y[train_idx])

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y[train_idx])

            predictions[test_idx] = 0.7 * gb.predict(X_test) + 0.3 * ridge.predict(X_test)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        return {'r2': r2, 'n_runs': len(runs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', default='../../training-data')
    parser.add_argument('--shared_data', default='../../shared-data')
    args = parser.parse_args()

    loader = TimeSeriesRewardLoader(args.training_data, args.shared_data)
    predictor = CutoffPredictor()

    cutoffs = [100000, 200000, 300000, 400000, 500000, 600000, 800000, 1000000]

    print("\nCutoff Analysis: Predicting Final Reward from Early Training Data\n")
    print(f"{'Cutoff':>12} {'Runs':>8} {'R2':>10}")

    for cutoff in cutoffs:
        runs = loader.load(cutoff)
        if len(runs) < 10:
            continue

        result = predictor.evaluate(runs)
        if result:
            print(f"{cutoff:>12,} {result['n_runs']:>8} {result['r2']:>10.2f}")


if __name__ == "__main__":
    main()
