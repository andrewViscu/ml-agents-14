# Data loading utilities

import pandas as pd
import numpy as np
import os
from glob import glob


# Common column name variations
COLUMN_MAPPINGS = {
    'step': ['step', 'steps', 'Step', 'training_step'],
    'reward': ['mean_reward', 'reward', 'mean_group_reward', 'cumulative_reward'],
    'run_id': ['run_id', 'RunID', 'run', 'experiment_id'],
    'environment': ['environment', 'env', 'game_name'],
    'algorithm': ['trainer_type', 'algorithm', 'algo'],
}


def load_csv(path):
    """Load a CSV file."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df.fillna(0)
    except:
        return None


def load_directory(data_dir, pattern=None):
    """Load all CSV files from a directory into one dataframe."""
    if not os.path.exists(data_dir):
        return None

    files = glob(os.path.join(data_dir, "*.csv"))
    if pattern:
        files = [f for f in files if pattern.lower() in f.lower()]

    if not files:
        return None

    dfs = []
    for path in files:
        df = load_csv(path)
        if df is not None:
            if 'run_id' not in df.columns:
                df['run_id'] = os.path.splitext(os.path.basename(path))[0]
            dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def find_column(df, standard_name):
    """Find a column by checking common name variations."""
    if standard_name not in COLUMN_MAPPINGS:
        return None

    cols_lower = {c.lower(): c for c in df.columns}
    for variation in COLUMN_MAPPINGS[standard_name]:
        if variation.lower() in cols_lower:
            return cols_lower[variation.lower()]
    return None


def normalize_columns(df):
    """Rename columns to standard names and add reward_metric column."""
    result = df.copy()

    # Rename known columns to standard names
    for standard, variations in COLUMN_MAPPINGS.items():
        actual = find_column(df, standard)
        if actual and actual != standard:
            result = result.rename(columns={actual: standard})

    # Create reward_metric per-run (different envs use different columns)
    result['reward_metric'] = 0.0

    # Priority order for reward columns
    reward_cols = ['mean_group_reward', 'cumulative_reward', 'mean_reward', 'reward']
    available_reward_cols = [c for c in reward_cols if c in result.columns]

    if 'run_id' in result.columns and available_reward_cols:
        for run_id in result['run_id'].unique():
            mask = result['run_id'] == run_id
            run_data = result.loc[mask]

            # Find first reward column with variance for this run
            for col in available_reward_cols:
                vals = run_data[col].values
                if np.std(vals) > 1e-6:
                    result.loc[mask, 'reward_metric'] = run_data[col].values
                    break
    elif available_reward_cols:
        # No run_id, just use first available with variance
        for col in available_reward_cols:
            if np.std(result[col].values) > 1e-6:
                result['reward_metric'] = result[col]
                break

    return result


def get_metadata(df):
    """Get basic info about the data."""
    info = {}

    if 'run_id' in df.columns:
        info['n_runs'] = df['run_id'].nunique()
    else:
        info['n_runs'] = 1

    if 'step' in df.columns:
        info['max_step'] = int(df['step'].max())

    if 'environment' in df.columns:
        info['environments'] = df['environment'].unique().tolist()

    return info


def aggregate_run(df, run_id):
    """Convert a time-series run into a summary row."""
    if len(df) == 0:
        return None

    summary = {'run_id': run_id}

    # Environment detection
    if 'environment' in df.columns:
        summary['environment'] = df['environment'].iloc[0]
    elif 'SOCCER' in run_id.upper():
        summary['environment'] = 'SoccerTwos'
    elif 'WORM' in run_id.upper():
        summary['environment'] = 'Worm'
    else:
        summary['environment'] = 'Unknown'

    # Algorithm
    if 'algorithm' in df.columns:
        summary['algorithm'] = df['algorithm'].iloc[0]

    # Hyperparameters
    hp_cols = ['learning_rate', 'batch_size', 'buffer_size', 'num_epoch']
    for col in hp_cols:
        if col in df.columns:
            summary[col] = df[col].iloc[0]

    # Training progress
    if 'step' in df.columns:
        summary['total_steps'] = df['step'].max()

    # Resource usage
    resource_cols = {
        'memory_usage_peak_mb': 'peak_ram_mb',
        'cpu_usage_avg_percent': 'avg_cpu_percent',
        'gpu_usage_avg_mb': 'avg_gpu_mb'
    }
    for src, dst in resource_cols.items():
        if src in df.columns:
            summary[dst] = df[src].mean()

    return summary


def load_resource_data(timeseries_dir=None, summary_dir=None):
    """Load resource data from time-series and/or summary format."""
    dfs = []

    # Load time-series data
    if timeseries_dir and os.path.exists(timeseries_dir):
        files = glob(os.path.join(timeseries_dir, "*.csv"))
        summaries = []

        for path in files:
            df = load_csv(path)
            if df is not None:
                run_id = os.path.splitext(os.path.basename(path))[0]
                row = aggregate_run(df, run_id)
                if row:
                    summaries.append(row)

        if summaries:
            dfs.append(pd.DataFrame(summaries))

    # Load summary data
    if summary_dir and os.path.exists(summary_dir):
        files = glob(os.path.join(summary_dir, "*.csv"))

        for path in files:
            df = load_csv(path)
            if df is not None:
                df = df.rename(columns={'enviroment': 'environment'})
                dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    return combined.fillna(0)


def extract_features(df, target_cols):
    """
    Extract features and targets for resource prediction.
    Returns (X, Y, feature_names, target_names)
    """
    feature_cols = []

    # One-hot encode environment
    if 'environment' in df.columns:
        env_dummies = pd.get_dummies(df['environment'], prefix='env')
        feature_cols.extend(env_dummies.columns.tolist())
        df = pd.concat([df, env_dummies], axis=1)

    # One-hot encode algorithm
    if 'algorithm' in df.columns:
        algo_dummies = pd.get_dummies(df['algorithm'], prefix='algo')
        feature_cols.extend(algo_dummies.columns.tolist())
        df = pd.concat([df, algo_dummies], axis=1)

    # Add numeric hyperparameters as features
    hp_cols = ['learning_rate', 'batch_size', 'buffer_size', 'num_epoch',
               'hidden_units', 'num_layers', 'time_horizon', 'total_steps']
    for col in hp_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Find which target columns exist
    actual_targets = [c for c in target_cols if c in df.columns]

    if not feature_cols or not actual_targets:
        return None, None, [], []

    X = df[feature_cols].values.astype(np.float64)
    Y = df[actual_targets].values.astype(np.float64)

    return X, Y, feature_cols, actual_targets
