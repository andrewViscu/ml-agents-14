import pandas as pd
import numpy as np
import os
from glob import glob


class SoccerTwosLoader:

    def __init__(self, shared_data_dir):
        self.shared_data_dir = shared_data_dir

    def load(self, min_steps=500000):
        csv_path = os.path.join(self.shared_data_dir, "multi_run", "only_soccertwo.csv")
        if not os.path.exists(csv_path):
            return []

        df = pd.read_csv(csv_path)
        df = df[df['total_steps'] >= min_steps]

        runs = []
        for _, row in df.iterrows():
            run = {
                'run_id': row['run_id'],
                'environment': 'SoccerTwos',
                'algorithm': row['algorithm'].upper(),
                'learning_rate': row['learning_rate'],
                'batch_size': row.get('bath_size', row.get('batch_size', 0)),
                'buffer_size': row['buffer_size'],
                'num_epoch': row['epoch'],
                'lambd': row['lambda'],
                'gamma': row.get('gamma', 0.99),
                'total_steps': row['total_steps'],
                'target_reward': row['mean_group_reward'],
            }

            if pd.notna(run['target_reward']):
                runs.append(run)

        return runs


class WormLoader:

    def __init__(self, shared_data_dir, training_data_dir):
        self.shared_data_dir = shared_data_dir
        self.training_data_dir = training_data_dir

    def load(self):
        runs = []
        runs.extend(self._load_from_single_run())
        runs.extend(self._load_from_training_data())
        return self._deduplicate(runs)

    def _load_from_single_run(self):
        static_path = os.path.join(self.shared_data_dir, "single_run", "static.csv")
        summary_path = os.path.join(self.shared_data_dir, "single_run", "summary.csv")

        if not os.path.exists(static_path) or not os.path.exists(summary_path):
            return []

        static_df = pd.read_csv(static_path)
        summary_df = pd.read_csv(summary_path)
        merged = pd.merge(static_df, summary_df, on='RunID', how='inner')
        worm_df = merged[merged['game_name'].str.lower().str.contains('worm', na=False)]

        runs = []
        for _, row in worm_df.iterrows():
            run = {
                'run_id': row['RunID'],
                'environment': 'Worm',
                'algorithm': row.get('training_type', 'ppo').upper(),
                'learning_rate': row.get('learning_rate', 0),
                'batch_size': row.get('batch_size', 0),
                'buffer_size': row.get('buffer_size', 0),
                'num_epoch': row.get('num_epoch', 0),
                'lambd': row.get('lambd', 0.95),
                'gamma': row.get('gamma', 0.99),
                'total_steps': row.get('max_steps', 0),
                'target_reward': row.get('final_reward', np.nan),
            }

            if pd.notna(run['target_reward']):
                runs.append(run)

        return runs

    def _load_from_training_data(self):
        pattern = os.path.join(self.training_data_dir, "*worm*.csv")
        files = [f for f in glob(pattern, recursive=False) if 'hardware' not in f.lower()]

        pattern_upper = os.path.join(self.training_data_dir, "*WORM*.csv")
        files.extend([f for f in glob(pattern_upper, recursive=False) if 'hardware' not in f.lower()])
        files = list(set(files))

        runs = []
        for filepath in files:
            run = self._load_training_file(filepath)
            if run:
                runs.append(run)

        return runs

    def _load_training_file(self, filepath):
        try:
            df = pd.read_csv(filepath)
            run_id = os.path.splitext(os.path.basename(filepath))[0]

            if 'mean_reward' not in df.columns or 'step' not in df.columns:
                return None

            max_step = df['step'].max()
            final_df = df[df['step'] >= max_step * 0.8]
            final_reward = np.percentile(final_df['mean_reward'].dropna(), 90) if len(final_df) > 0 else df['mean_reward'].iloc[-1]

            algorithm = 'PPO'
            if 'trainer_type' in df.columns:
                trainer = str(df['trainer_type'].iloc[0]).lower()
                if 'sac' in trainer:
                    algorithm = 'SAC'

            return {
                'run_id': run_id,
                'environment': 'Worm',
                'algorithm': algorithm,
                'learning_rate': df['learning_rate'].iloc[0] if 'learning_rate' in df.columns else 0,
                'batch_size': df['batch_size'].iloc[0] if 'batch_size' in df.columns else 0,
                'buffer_size': df['buffer_size'].iloc[0] if 'buffer_size' in df.columns else 0,
                'num_epoch': df['num_epoch'].iloc[0] if 'num_epoch' in df.columns else 0,
                'lambd': df['lambd'].iloc[0] if 'lambd' in df.columns else 0.95,
                'gamma': 0.99,
                'total_steps': max_step,
                'target_reward': final_reward,
            }
        except Exception:
            return None

    def _deduplicate(self, runs):
        seen = set()
        unique = []
        for run in runs:
            if run['run_id'] not in seen:
                unique.append(run)
                seen.add(run['run_id'])
        return unique


class RQ2DataLoader:

    def __init__(self, training_data_dir, shared_data_dir):
        self.soccer_loader = SoccerTwosLoader(shared_data_dir)
        self.worm_loader = WormLoader(shared_data_dir, training_data_dir)

    def load(self, min_steps=500000):
        return {
            'soccertwos': self.soccer_loader.load(min_steps),
            'worm': self.worm_loader.load()
        }
