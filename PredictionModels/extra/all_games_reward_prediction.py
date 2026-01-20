import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MULTI_AGENT_GAMES = ['SoccerTwos', 'PushBlock', 'Hallway', 'GridFoodCollector']
SINGLE_AGENT_GAMES = ['3DBall', '3DBallHard', 'Crawler', 'Worm', 'Walker', 'Basic', 'BigWallJump', 'Pyramids', 'Sorter', 'GridWorld']


class AllGamesDataLoader:

    def __init__(self, shared_data_dir):
        self.shared_data_dir = shared_data_dir

    def load(self):
        all_runs = {}

        self._load_single_run_data(all_runs)
        self._load_multiple_games_lots(all_runs)
        self._load_soccertwos(all_runs)

        return all_runs

    def _load_single_run_data(self, all_runs):
        static_path = os.path.join(self.shared_data_dir, "single_run", "static.csv")
        summary_path = os.path.join(self.shared_data_dir, "single_run", "summary.csv")

        if not os.path.exists(static_path) or not os.path.exists(summary_path):
            return

        static_df = pd.read_csv(static_path)
        summary_df = pd.read_csv(summary_path)
        merged = pd.merge(static_df, summary_df, on='RunID', how='inner')

        for _, row in merged.iterrows():
            game = row.get('game_name', '')
            if not game or pd.isna(game):
                continue

            game = self._normalize_game_name(game)
            reward = row.get('final_reward', np.nan)

            if pd.isna(reward):
                continue

            run = {
                'run_id': row['RunID'],
                'algorithm': str(row.get('training_type', 'ppo')).upper(),
                'learning_rate': row.get('learning_rate', 0),
                'batch_size': row.get('batch_size', 0),
                'buffer_size': row.get('buffer_size', 0),
                'num_epoch': row.get('num_epoch', 0),
                'lambd': row.get('lambd', 0.95),
                'total_steps': row.get('max_steps', 0),
                'target_reward': reward,
            }

            if game not in all_runs:
                all_runs[game] = []
            all_runs[game].append(run)

    def _load_multiple_games_lots(self, all_runs):
        csv_path = os.path.join(self.shared_data_dir, "multi_run", "multiple_games_lots.csv")

        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            game = row.get('environment', '')
            if not game or pd.isna(game):
                continue

            game = self._normalize_game_name(game)
            reward = row.get('reward_mean', np.nan)

            if pd.isna(reward):
                continue

            run = {
                'run_id': row.get('run_id', ''),
                'algorithm': str(row.get('algorithm', 'ppo')).upper(),
                'learning_rate': row.get('learning_rate', 0),
                'batch_size': row.get('batch_size', 0),
                'buffer_size': row.get('buffer_size', 0),
                'num_epoch': row.get('epochs', 0),
                'lambd': 0.95,
                'total_steps': row.get('steps', 0),
                'target_reward': reward,
            }

            if game not in all_runs:
                all_runs[game] = []
            all_runs[game].append(run)

    def _load_soccertwos(self, all_runs):
        csv_path = os.path.join(self.shared_data_dir, "multi_run", "only_soccertwo.csv")

        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        df = df[df['total_steps'] >= 500000]

        game = 'SoccerTwos'
        if game not in all_runs:
            all_runs[game] = []

        for _, row in df.iterrows():
            reward = row.get('mean_group_reward', np.nan)
            if pd.isna(reward):
                continue

            run = {
                'run_id': row['run_id'],
                'algorithm': str(row.get('algorithm', 'ppo')).upper(),
                'learning_rate': row.get('learning_rate', 0),
                'batch_size': row.get('bath_size', row.get('batch_size', 0)),
                'buffer_size': row.get('buffer_size', 0),
                'num_epoch': row.get('epoch', 0),
                'lambd': row.get('lambda', 0.95),
                'total_steps': row.get('total_steps', 0),
                'target_reward': reward,
            }
            all_runs[game].append(run)

    def _normalize_game_name(self, name):
        name = str(name).strip()
        name_lower = name.lower()

        mappings = {
            '3dball': '3DBall',
            '3dballhard': '3DBallHard',
            'crawler': 'Crawler',
            'worm': 'Worm',
            'walker': 'Walker',
            'pushblock': 'PushBlock',
            'hallway': 'Hallway',
            'pyramids': 'Pyramids',
            'sorter': 'Sorter',
            'gridworld': 'GridWorld',
            'gridfoodcollector': 'GridFoodCollector',
            'soccertwos': 'SoccerTwos',
            'basic': 'Basic',
            'bigwalljump': 'BigWallJump',
        }

        for key, val in mappings.items():
            if key in name_lower:
                return val

        return name


class DataValidator:

    @staticmethod
    def check_hp_variation(runs, min_runs=30):
        if len(runs) < min_runs:
            return False, 'insufficient_samples'

        hp_keys = ['learning_rate', 'batch_size', 'buffer_size', 'num_epoch']
        unique_configs = set()

        for run in runs:
            config = tuple(run.get(k, 0) for k in hp_keys)
            unique_configs.add(config)

        if len(unique_configs) < 3:
            return False, 'no_hp_variation'

        rewards = [r['target_reward'] for r in runs]
        reward_std = np.std(rewards)

        if reward_std < 0.01:
            return False, 'no_reward_variation'

        return True, 'ok'

    @staticmethod
    def get_hp_stats(runs):
        hp_keys = ['learning_rate', 'batch_size', 'buffer_size', 'num_epoch']
        stats = {}

        for key in hp_keys:
            values = [r.get(key, 0) for r in runs]
            unique = len(set(values))
            stats[key] = unique

        rewards = [r['target_reward'] for r in runs]
        stats['reward_range'] = max(rewards) - min(rewards)
        stats['reward_std'] = np.std(rewards)

        return stats


class RewardPredictor:

    FEATURE_NAMES = [
        'learning_rate', 'log_learning_rate', 'batch_size', 'log_batch_size',
        'buffer_size', 'log_buffer_size', 'num_epoch', 'lambd',
        'total_steps', 'log_total_steps', 'batch_buffer_ratio', 'effective_lr',
        'algo_PPO', 'algo_SAC', 'algo_POCA'
    ]

    def extract_features(self, runs):
        X_data = []
        y = []

        for run in runs:
            lr = run.get('learning_rate', 0) or 1e-6
            batch = run.get('batch_size', 0) or 1
            buffer = run.get('buffer_size', 0) or 1
            steps = run.get('total_steps', 0) or 1
            algo = run.get('algorithm', 'PPO').upper()

            features = {
                'learning_rate': lr,
                'log_learning_rate': np.log10(max(lr, 1e-8)),
                'batch_size': batch,
                'log_batch_size': np.log2(max(batch, 1)),
                'buffer_size': buffer,
                'log_buffer_size': np.log2(max(buffer, 1)),
                'num_epoch': run.get('num_epoch', 0) or 0,
                'lambd': run.get('lambd', 0.95) or 0.95,
                'total_steps': steps,
                'log_total_steps': np.log10(max(steps, 1)),
                'batch_buffer_ratio': batch / max(buffer, 1),
                'effective_lr': lr * batch,
                'algo_PPO': 1 if algo == 'PPO' else 0,
                'algo_SAC': 1 if algo == 'SAC' else 0,
                'algo_POCA': 1 if algo == 'POCA' else 0,
            }

            X_data.append([features[name] for name in self.FEATURE_NAMES])
            y.append(run['target_reward'])

        X = np.array(X_data)
        y = np.array(y)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        return X, y

    def evaluate(self, runs, n_folds=5):
        if len(runs) < 10:
            return None

        X, y = self.extract_features(runs)
        n_folds = min(n_folds, len(runs) // 2)

        if n_folds < 2:
            return None

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        predictions = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            )
            gb.fit(X_train, y[train_idx])

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y[train_idx])

            predictions[test_idx] = 0.7 * gb.predict(X_test) + 0.3 * ridge.predict(X_test)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        mae = np.mean(np.abs(y - predictions))
        rel_err = np.abs((y - predictions) / (np.abs(y) + 0.01)) * 100
        within_25 = np.mean(rel_err <= 25) * 100
        within_50 = np.mean(rel_err <= 50) * 100

        reward_range = y.max() - y.min()
        reward_std = y.std()

        return {
            'r2': r2,
            'mae': mae,
            'within_25': within_25,
            'within_50': within_50,
            'reward_range': reward_range,
            'reward_std': reward_std,
            'n_runs': len(runs)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared_data', default='../../shared-data')
    parser.add_argument('--show_excluded', action='store_true')
    args = parser.parse_args()

    loader = AllGamesDataLoader(args.shared_data)
    predictor = RewardPredictor()
    validator = DataValidator()

    all_runs = loader.load()

    excluded = []
    valid_games = {}

    for game, runs in all_runs.items():
        is_valid, reason = validator.check_hp_variation(runs)
        if is_valid:
            valid_games[game] = runs
        else:
            stats = validator.get_hp_stats(runs) if len(runs) > 0 else {}
            excluded.append((game, len(runs), reason, stats))

    print("\nReward Prediction by Game (from hyperparameters)\n")

    print("Single-Agent Environments:")
    print(f"{'Game':<20} {'Runs':>6} {'R2':>8} {'MAE':>10} {'<25%':>8} {'<50%':>8} {'Range':>10}")

    single_results = []
    for game in sorted(valid_games.keys()):
        if game not in SINGLE_AGENT_GAMES:
            continue
        runs = valid_games[game]
        result = predictor.evaluate(runs)
        if result:
            single_results.append((game, result))
            print(f"{game:<20} {result['n_runs']:>6} {result['r2']:>8.2f} {result['mae']:>10.2f} "
                  f"{result['within_25']:>7.1f}% {result['within_50']:>7.1f}% {result['reward_range']:>10.1f}")

    print("\nMulti-Agent Environments:")
    print(f"{'Game':<20} {'Runs':>6} {'R2':>8} {'MAE':>10} {'<25%':>8} {'<50%':>8} {'Range':>10}")

    multi_results = []
    for game in sorted(valid_games.keys()):
        if game not in MULTI_AGENT_GAMES:
            continue
        runs = valid_games[game]
        result = predictor.evaluate(runs)
        if result:
            multi_results.append((game, result))
            print(f"{game:<20} {result['n_runs']:>6} {result['r2']:>8.2f} {result['mae']:>10.2f} "
                  f"{result['within_25']:>7.1f}% {result['within_50']:>7.1f}% {result['reward_range']:>10.1f}")

    if excluded:
        print(f"\nExcluded Games ({len(excluded)}):")
        print(f"{'Game':<20} {'Runs':>6} {'Reason':<25} {'Details'}")
        for game, n_runs, reason, stats in sorted(excluded):
            if reason == 'insufficient_samples':
                detail = f"need >= 30 runs"
            elif reason == 'no_hp_variation':
                detail = f"only {stats.get('learning_rate', 0)} LR, {stats.get('batch_size', 0)} batch configs"
            elif reason == 'no_reward_variation':
                detail = f"reward std={stats.get('reward_std', 0):.4f}"
            else:
                detail = ""
            print(f"{game:<20} {n_runs:>6} {reason:<25} {detail}")

    print("\nSummary:")
    if single_results:
        avg_r2_single = np.mean([r['r2'] for _, r in single_results])
        total_single = sum(r['n_runs'] for _, r in single_results)
        print(f"  Single-Agent: {len(single_results)} games, {total_single} runs, Avg R2: {avg_r2_single:.2f}")

    if multi_results:
        avg_r2_multi = np.mean([r['r2'] for _, r in multi_results])
        total_multi = sum(r['n_runs'] for _, r in multi_results)
        print(f"  Multi-Agent: {len(multi_results)} games, {total_multi} runs, Avg R2: {avg_r2_multi:.2f}")

    print("\nKey Observations:")
    all_results = single_results + multi_results
    for game, result in sorted(all_results, key=lambda x: x[1]['r2'], reverse=True):
        if result['r2'] > 0.4:
            print(f"  {game}: R2={result['r2']:.2f} - good predictability ({result['n_runs']} runs, range={result['reward_range']:.1f})")
    for game, result in sorted(all_results, key=lambda x: x[1]['r2']):
        if result['r2'] < 0.1:
            print(f"  {game}: R2={result['r2']:.2f} - poor predictability (range={result['reward_range']:.1f}, likely too narrow)")


if __name__ == "__main__":
    main()
