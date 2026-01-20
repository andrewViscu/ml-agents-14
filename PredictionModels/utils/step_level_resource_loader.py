import pandas as pd
import numpy as np
import os
from glob import glob


class StepLevelResourceLoader:

    RESOURCE_COLUMNS = ['memory_usage_avg_mb', 'cpu_usage_avg_percent', 'gpu_usage_avg_mb']
    HYPERPARAM_COLUMNS = ['batch_size', 'buffer_size', 'learning_rate', 'beta',
                          'epsilon', 'lambd', 'num_epoch', 'trainer_type']

    def __init__(self, training_data_dir):
        self.training_data_dir = training_data_dir

    def load(self, games=None):
        if games is None:
            games = ['soccertwos', 'worm']

        games_lower = [g.lower() for g in games]
        all_data = []
        files = [f for f in glob(os.path.join(self.training_data_dir, "*.csv"))
                 if 'hardware' not in f.lower()]

        for filepath in files:
            filename = os.path.basename(filepath).upper()
            game = self._detect_game(filename)

            if game is None or game.lower() not in games_lower:
                continue

            records = self._load_file(filepath, game)
            all_data.extend(records)

        if not all_data:
            return pd.DataFrame()

        return pd.DataFrame(all_data)

    def _detect_game(self, filename):
        if 'SOCCER' in filename:
            return 'SoccerTwos'
        elif 'WORM' in filename:
            return 'Worm'
        return None

    def _load_file(self, filepath, game):
        records = []
        try:
            df = pd.read_csv(filepath)
            if 'step' not in df.columns:
                return records

            available_resources = [c for c in self.RESOURCE_COLUMNS if c in df.columns]
            if not available_resources:
                return records

            run_id = os.path.splitext(os.path.basename(filepath))[0]
            hyperparams = {col: df[col].iloc[0] if col in df.columns else 0
                          for col in self.HYPERPARAM_COLUMNS}
            algorithm = self._detect_algorithm(df)

            for _, row in df.iterrows():
                record = {
                    'run_id': run_id,
                    'game': game,
                    'algorithm': algorithm,
                    'step': row['step'],
                    'memory_usage_avg_mb': row.get('memory_usage_avg_mb', np.nan),
                    'cpu_usage_avg_percent': row.get('cpu_usage_avg_percent', np.nan),
                    'gpu_usage_avg_mb': row.get('gpu_usage_avg_mb', np.nan),
                    'batch_size': hyperparams.get('batch_size', 0),
                    'buffer_size': hyperparams.get('buffer_size', 0),
                    'learning_rate': hyperparams.get('learning_rate', 0),
                    'beta': hyperparams.get('beta', 0),
                    'epsilon': hyperparams.get('epsilon', 0),
                    'lambd': hyperparams.get('lambd', 0.95),
                    'num_epoch': hyperparams.get('num_epoch', 0),
                }
                records.append(record)
        except Exception:
            pass

        return records

    def _detect_algorithm(self, df):
        if 'trainer_type' in df.columns:
            trainer = str(df['trainer_type'].iloc[0]).lower()
            if 'sac' in trainer:
                return 'SAC'
            elif 'poca' in trainer:
                return 'POCA'
        return 'PPO'


class FeatureExtractor:

    FEATURE_NAMES = [
        'step', 'log_step', 'batch_size', 'log_batch_size', 'buffer_size',
        'log_buffer_size', 'learning_rate', 'log_learning_rate', 'beta',
        'epsilon', 'lambd', 'num_epoch', 'batch_buffer_ratio',
        'game_SoccerTwos', 'game_Worm', 'algo_PPO', 'algo_SAC', 'algo_POCA'
    ]

    def extract(self, df, target_col):
        df = df.dropna(subset=[target_col])
        if len(df) == 0:
            return None, None, None

        X_data = []
        for _, row in df.iterrows():
            features = self._extract_row_features(row)
            X_data.append(features)

        X = np.array([[f[name] for name in self.FEATURE_NAMES] for f in X_data])
        y = df[target_col].values

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        metadata = {
            'feature_names': self.FEATURE_NAMES,
            'run_ids': df['run_id'].values,
            'games': df['game'].values,
        }

        return X, y, metadata

    def _extract_row_features(self, row):
        return {
            'step': row['step'],
            'log_step': np.log10(max(row['step'], 1)),
            'batch_size': row['batch_size'],
            'log_batch_size': np.log2(max(row['batch_size'], 1)),
            'buffer_size': row['buffer_size'],
            'log_buffer_size': np.log2(max(row['buffer_size'], 1)),
            'learning_rate': row['learning_rate'],
            'log_learning_rate': np.log10(max(row['learning_rate'], 1e-8)),
            'beta': row['beta'],
            'epsilon': row['epsilon'],
            'lambd': row['lambd'],
            'num_epoch': row['num_epoch'],
            'batch_buffer_ratio': row['batch_size'] / max(row['buffer_size'], 1),
            'game_SoccerTwos': 1 if row['game'] == 'SoccerTwos' else 0,
            'game_Worm': 1 if row['game'] == 'Worm' else 0,
            'algo_PPO': 1 if row['algorithm'] == 'PPO' else 0,
            'algo_SAC': 1 if row['algorithm'] == 'SAC' else 0,
            'algo_POCA': 1 if row['algorithm'] == 'POCA' else 0,
        }
