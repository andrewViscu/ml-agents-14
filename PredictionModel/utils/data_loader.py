#utility functions for loading and preprocessing ml-agents training data

import pandas as pd
import numpy as np
import os


def load_single_file(file_path):
    #loads a single csv file and returns dataframe
    #handles both timestamp and time_elapsed formats
    
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    return df


def load_training_data(data_dir, file_pattern=None):
    #loads all csv files from a directory and combines them
    #file_pattern can be used to filter specific files like 'ppo' or 'sac'
    
    all_data = []
    
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
        if file_pattern and file_pattern.lower() not in file.lower():
            continue
            
        path = os.path.join(data_dir, file)
        df = load_single_file(path)
        all_data.append(df)
    
    if len(all_data) == 0:
        print(f"Warning: No csv files found in {data_dir}")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(all_data)} files, {len(combined)} total samples")
    return combined


def load_multiple_files(file_paths):
    #loads specific files from a list of paths
    
    all_data = []
    for path in file_paths:
        if os.path.exists(path):
            df = load_single_file(path)
            all_data.append(df)
        else:
            print(f"Warning: {path} not found, skipping")
    
    if len(all_data) == 0:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def extract_features(df, feature_cols=None):
    #extracts feature columns from dataframe
    #returns X matrix and list of feature names used
    
    if feature_cols is None:
        #default features - resource usage metrics
        feature_cols = [
            'step',
            'memory_usage_avg_mb',
            'memory_usage_peak_mb',
            'cpu_usage_avg_mb',
            'cpu_usage_peak_mb',
            'gpu_usage_avg_mb',
            'gpu_usage_peak_mb'
        ]
    
    #only use columns that actually exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("Error: No valid feature columns found!")
        return None, []
    
    X = df[available_cols].values
    return X, available_cols


def create_reward_labels(df, threshold=0.05):
    #binary labels based on reward threshold
    #1 = good performance, 0 = bad performance
    
    labels = (df['mean_group_reward'] > threshold).astype(int).values
    return labels


def create_algorithm_labels(df):
    #creates labels for algorithm classification
    #maps algorithm names to integers
    
    algorithm_map = {
        'ppo': 0,
        'sac': 1,
        'poca': 2
    }
    
    #handle case sensitivity
    algorithms = df['trainer_type'].str.lower()
    labels = algorithms.map(algorithm_map).fillna(-1).astype(int).values
    
    return labels, algorithm_map


def get_algorithm_data(df, algorithm_name):
    #filters dataframe to only include specific algorithm
    
    mask = df['trainer_type'].str.lower() == algorithm_name.lower()
    return df[mask].copy()


def split_by_run(df):
    #splits data by run_id for proper train/test splitting
    #returns dict of run_id -> dataframe
    
    runs = {}
    for run_id in df['run_id'].unique():
        runs[run_id] = df[df['run_id'] == run_id].copy()
    return runs


def compute_run_statistics(df):
    #computes aggregate statistics for each run
    #useful for run-level predictions instead of step-level
    
    stats = df.groupby('run_id').agg({
        'mean_group_reward': ['mean', 'std', 'max', 'min'],
        'memory_usage_avg_mb': ['mean', 'max'],
        'cpu_usage_avg_mb': ['mean', 'max'],
        'step': 'max'
    })
    
    #flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    
    return stats
