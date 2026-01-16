# Resource prediction CLI
# Predicts training time, CPU, RAM usage from configuration

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_resource_data, extract_features
from models.resource_model import ResourceModel


# Target columns to predict
TARGETS = [
    'total_time', 'training_time', 'time_elapsed_seconds',
    'average_cpu', 'avg_cpu_percent', 'cpu_usage_percent',
    'average_ram', 'avg_ram_mb', 'ram_usage_mb'
]


def train(our_data, shared_data):
    df = load_resource_data(our_data, shared_data)
    if df is None:
        return None

    X, Y, feature_names, target_names = extract_features(df, TARGETS)

    if X is None or len(feature_names) == 0 or len(target_names) == 0:
        return None

    model = ResourceModel()
    results = model.train(X, Y, feature_names, target_names)

    if results is None:
        return None

    for target, r2 in results.items():
        print(f"{target}: R2={r2:.4f}")

    return model, X, Y, feature_names, target_names


def evaluate(our_data, shared_data):
    result = train(our_data, shared_data)
    if result is None:
        return

    model, X, Y, feature_names, target_names = result

    for i, target in enumerate(target_names):
        print(f"\n{target}:")
        for j in range(min(10, len(X))):
            preds = model.predict(X[j])
            if target in preds:
                actual = Y[j, i]
                predicted = preds[target]
                if abs(actual) > 0.01:
                    error = abs(predicted - actual) / abs(actual) * 100
                    print(f"  {j+1}. actual={actual:.2f} pred={predicted:.2f} err={error:.1f}%")
                else:
                    print(f"  {j+1}. actual={actual:.2f} pred={predicted:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Predict resource requirements')
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--our_data', default='../training-data')
    parser.add_argument('--shared_data', default='../shared-data/multi_run')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.our_data, args.shared_data)
    elif args.command == 'evaluate':
        evaluate(args.our_data, args.shared_data)


if __name__ == "__main__":
    main()
