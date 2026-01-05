#main entry point for ml-agents prediction project
#Group 14 - BCS2720 Project 2.1
#
#usage:
#   python main.py                     - runs with default data directory
#   python main.py --data_dir ./data   - specify data directory
#   python main.py --mode reward       - predict reward performance
#   python main.py --mode algorithm    - predict best algorithm

import argparse
import numpy as np

from utils.data_loader import (
    load_training_data,
    load_multiple_files,
    extract_features,
    create_reward_labels,
    create_algorithm_labels
)

from models.random_forest import MyRandomForest, cross_validation, analyze_feature_importance
from models.algorithm_predictor import (
    AlgorithmPredictor,
    EnvironmentAlgorithmPredictor,
    compare_algorithms,
    predict_convergence
)


def run_reward_prediction(df, threshold=0.05):
    #predicts if a training step will have good or bad reward
    
    print("\n" + "="*50)
    print("REWARD PREDICTION")
    print("="*50)
    
    X, feature_names = extract_features(df)
    y = create_reward_labels(df, threshold=threshold)
    
    print(f"\nFeatures: {feature_names}")
    print(f"Samples: {len(y)}")
    print(f"Class distribution:")
    print(f"  Bad reward (0): {(y == 0).sum()}")
    print(f"  Good reward (1): {(y == 1).sum()}")
    
    #cross validation
    print("\n--- Cross Validation ---")
    mean_score, scores = cross_validation(X, y, k=5, model_class=MyRandomForest, n_trees=50)
    print(f"Average CV Score: {mean_score:.3f}")
    
    #train final model
    print("\n--- Final Model ---")
    model = MyRandomForest(n_trees=100)
    model.fit(X, y)
    
    train_preds = model.predict(X)
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y, train_preds)
    print(f"Training accuracy: {train_acc:.3f}")
    
    #feature importance
    importance = analyze_feature_importance(model, feature_names)
    print("\nFeature Importance:")
    for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.3f}")
    
    return model, mean_score


def run_algorithm_prediction(df):
    # predicts which algorithm is being used based on training metrics
    # this framework can be extended to predict best algorithm for environment

    print("\n" + "=" * 50)
    print("ALGORITHM PREDICTION")
    print("=" * 50)

    # show algorithm comparison first
    compare_algorithms(df)

    # check what algorithms we have
    algorithms = df['trainer_type'].unique()
    print(f"Algorithms in data: {list(algorithms)}")

    if len(algorithms) < 2:
        print("\n" + "-" * 40)
        print("SKIPPING ALGORITHM CLASSIFICATION")
        print("-" * 40)
        print("Need at least 2 different algorithms for comparison!")
        print(f"Currently only have: {list(algorithms)}")
        print("\nOnce you add PPO and SAC data, run this again for full comparison.")
        print("The framework is ready - just add more training runs.")
        return None, None

    # create predictor
    predictor = AlgorithmPredictor(n_trees=50)

    # prepare data (using early training metrics - first 20% of each run)
    X, y = predictor.prepare_data(df, early_steps_only=True, early_percent=0.2)

    print(f"\nSamples: {len(y)}")
    print(f"Algorithm distribution:")
    for algo, label in predictor.algorithm_map.items():
        count = (y == label).sum()
        if count > 0:
            print(f"  {algo}: {count}")

    # cross validation (will be skipped if only 1 class)
    result = predictor.cross_validate(X, y, k=5)
    if result[0] is None:
        return None, None

    mean_score, scores = result

    # train final model
    print("\n--- Final Model ---")
    predictor.train(X, y)

    # feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\nFeature Importance:")
        for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.3f}")

    return predictor, mean_score

def run_environment_analysis(df):
    #analyzes performance by environment
    #builds framework for recommending algorithms per environment
    
    print("\n" + "="*50)
    print("ENVIRONMENT ANALYSIS")
    print("="*50)
    
    env_predictor = EnvironmentAlgorithmPredictor()
    scores, best = env_predictor.prepare_recommendation_data(df)
    
    print("\nAlgorithm scores by environment:")
    print(scores.to_string())
    
    print("\nBest algorithm per environment:")
    print(best.to_string())
    
    return env_predictor


def run_convergence_analysis(df):
    #analyzes if training runs are converging
    
    print("\n" + "="*50)
    print("CONVERGENCE ANALYSIS")
    print("="*50)
    
    for algo in df['trainer_type'].unique():
        result, message = predict_convergence(df, algo)
        print(f"\n{algo.upper()}: {message}")


def main():
    parser = argparse.ArgumentParser(description='ML-Agents Training Data Predictor')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing training CSV files')
    parser.add_argument('--files', type=str, nargs='+', help='Specific files to load')
    parser.add_argument('--mode', type=str, default='all', choices=['reward', 'trainer_type', 'environment', 'convergence', 'all'],
                        help='Which analysis to run')
    parser.add_argument('--threshold', type=float, default=0.05, help='Reward threshold for classification')
    
    args = parser.parse_args()
    
    #load data
    print("Loading data...")
    
    if args.files:
        df = load_multiple_files(args.files)
    else:
        df = load_training_data(args.data_dir)
    
    if df is None or len(df) == 0:
        print("Error: No data loaded!")
        print(f"Make sure CSV files are in {args.data_dir} or specify files with --files")
        return
    
    print(f"Loaded {len(df)} samples")
    
    #run selected analyses
    if args.mode == 'reward' or args.mode == 'all':
        run_reward_prediction(df, threshold=args.threshold)
    
    if args.mode == 'trainer_type' or args.mode == 'all':
        run_algorithm_prediction(df)
    
    if args.mode == 'environment' or args.mode == 'all':
        run_environment_analysis(df)
    
    if args.mode == 'convergence' or args.mode == 'all':
        run_convergence_analysis(df)
    
    print("\n" + "="*50)
    print("DONE")
    print("="*50)


if __name__ == "__main__":
    main()
