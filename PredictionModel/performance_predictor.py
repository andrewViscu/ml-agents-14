#!/usr/bin/env python3
# Performance prediction CLI
# Predicts final performance from early training data

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_directory, normalize_columns
from models.performance_model import PerformanceModel


def train(data_dir, cutoff=500000):
    """Train the performance model."""
    df = load_directory(data_dir)
    if df is None:
        return None

    df = normalize_columns(df)

    # Train per-game models
    games = {}
    for run_id in df['run_id'].unique():
        if 'SOCCER' in run_id.upper():
            games.setdefault('SoccerTwos', []).append(run_id)
        elif 'WORM' in run_id.upper():
            games.setdefault('Worm', []).append(run_id)

    for game, runs in games.items():
        game_df = df[df['run_id'].isin(runs)]
        model = PerformanceModel(early_cutoff=cutoff)
        X, y, _ = model.prepare_data(game_df)
        if X is not None and len(X) >= 2:
            r2 = model.train(X, y)
            print(f"{game}: R2={r2:.4f} (n={len(X)})")

    # Train combined model
    model = PerformanceModel(early_cutoff=cutoff)
    X, y, run_ids = model.prepare_data(df)

    if X is None:
        print("Not enough data for training")
        return None

    r2 = model.train(X, y)
    print(f"Combined: R2={r2:.4f} (n={len(X)})")

    return model


def evaluate(data_dir, cutoff=500000):
    """Evaluate with per-run predictions."""
    df = load_directory(data_dir)
    if df is None:
        return

    df = normalize_columns(df)

    model = PerformanceModel(early_cutoff=cutoff)
    X, y, run_ids = model.prepare_data(df)

    if X is None:
        print("Not enough data")
        return

    model.train(X, y)

    for i, run_id in enumerate(run_ids):
        actual = y[i]
        pred = model.predict_ensemble(model.normalize(X[i:i+1], fit=False))[0]
        error = abs(pred - actual) / (abs(actual) + 0.001) * 100
        print(f"{run_id[:25]:<25} actual={actual:.2f} pred={pred:.2f} err={error:.1f}%")


def predict(file_path, data_dir, cutoff=500000):
    """Predict for a single file."""
    model = train(data_dir, cutoff)
    if model is None:
        return

    from utils.data_loader import load_csv
    df = load_csv(file_path)
    if df is None:
        print(f"Could not load {file_path}")
        return

    df = normalize_columns(df)

    result = model.predict(df)
    if result is None:
        print("Not enough data to predict")
    else:
        print(f"{result:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Predict final performance from early training')
    parser.add_argument('command', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--data_dir', default='../training-data')
    parser.add_argument('--file', help='CSV file for prediction')
    parser.add_argument('--cutoff', type=int, default=500000)

    args = parser.parse_args()

    if args.command == 'train':
        train(args.data_dir, args.cutoff)
    elif args.command == 'evaluate':
        evaluate(args.data_dir, args.cutoff)
    elif args.command == 'predict':
        if not args.file:
            print("--file required")
            return
        predict(args.file, args.data_dir, args.cutoff)


if __name__ == "__main__":
    main()
