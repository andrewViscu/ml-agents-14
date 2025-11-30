#algorithm predictor - predicts which algorithm (PPO, SAC, POCA) will perform best
#this is the main framework for comparing algorithm performance

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

#add parent dir to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.random_forest import MyRandomForest, cross_validation, analyze_feature_importance
from utils.data_loader import extract_features, create_algorithm_labels, compute_run_statistics


class AlgorithmPredictor:
    #predicts which algorithm will perform best for a given environment
    #uses early training metrics to make predictions
    
    def __init__(self, n_trees=50):
        self.n_trees = n_trees
        self.model = None
        self.algorithm_map = None
        self.reverse_map = None
        self.feature_names = None

    def prepare_data(self, df, early_steps_only=True, early_percent=0.2):
        # prepares data for algorithm prediction
        # if early_steps_only is True, only uses first X% of each run's data
        # this simulates predicting algorithm performance before full training completes

        if early_steps_only:
            # use percentage of each run instead of fixed step cutoff
            filtered_dfs = []
            for run_id in df['run_id'].unique():
                run_data = df[df['run_id'] == run_id]
                cutoff_idx = int(len(run_data) * early_percent)
                cutoff_idx = max(cutoff_idx, 10)  # at least 10 samples per run
                filtered_dfs.append(run_data.iloc[:cutoff_idx])

            df = pd.concat(filtered_dfs, ignore_index=True)
            print(f"Using early training data only (first {early_percent * 100:.0f}% of each run)")
        
        #get features
        X, self.feature_names = extract_features(df)
        
        #get algorithm labels
        y, self.algorithm_map = create_algorithm_labels(df)
        self.reverse_map = {v: k for k, v in self.algorithm_map.items()}
        
        #filter out unknown algorithms (label = -1)
        valid_mask = y >= 0
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def train(self, X, y):
        #trains the random forest on algorithm prediction
        
        self.model = MyRandomForest(n_trees=self.n_trees)
        self.model.fit(X, y)
        
        #compute training accuracy
        train_preds = self.model.predict(X)
        train_acc = accuracy_score(y, train_preds)
        print(f"Training accuracy: {train_acc:.3f}")
        
        return train_acc
    
    def predict(self, X):
        #predicts algorithm labels
        
        if self.model is None:
            print("Error: Model not trained yet!")
            return None
        
        return self.model.predict(X)
    
    def predict_algorithm_name(self, X):
        #predicts and returns algorithm names instead of labels
        
        predictions = self.predict(X)
        if predictions is None:
            return None
        
        names = [self.reverse_map.get(p, 'unknown') for p in predictions]
        return names
    
    def predict_proba(self, X):
        #returns probability for each algorithm
        
        if self.model is None:
            return None
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        #evaluates model on test data
        
        predictions = self.predict(X)
        acc = accuracy_score(y, predictions)
        
        #get class names for report
        target_names = [self.reverse_map[i] for i in sorted(self.reverse_map.keys()) if i in np.unique(y)]
        
        print(f"\nTest Accuracy: {acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=target_names, zero_division=0))
        
        return acc

    def cross_validate(self, X, y, k=5):
        # runs k-fold cross validation

        # check if we have enough samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"\nWarning: Only {len(unique_classes)} class(es) in data - CV is meaningless!")
            print("Need at least 2 different algorithms to do classification.")
            return None, None

        if len(y) < k * 2:
            print(f"\nWarning: Only {len(y)} samples for {k}-fold CV - results unreliable!")
            k = max(2, len(y) // 5)  # reduce k if not enough samples
            print(f"Reducing to {k}-fold CV")

        # check class balance
        for c in unique_classes:
            count = (y == c).sum()
            if count < k:
                print(f"Warning: Class {self.reverse_map.get(c, c)} only has {count} samples")

        print(f"\n--- {k}-Fold Cross Validation ---")
        mean_score, scores = cross_validation(X, y, k=k, model_class=MyRandomForest, n_trees=self.n_trees)
        print(f"Average CV Score: {mean_score:.3f}")
        return mean_score, scores
    
    def get_feature_importance(self):
        #returns feature importance dict
        
        if self.model is None or self.feature_names is None:
            return None
        
        return analyze_feature_importance(self.model, self.feature_names)


class EnvironmentAlgorithmPredictor:
    #higher level predictor that recommends best algorithm for an environment
    #aggregates run-level data instead of step-level
    
    def __init__(self, n_trees=50):
        self.n_trees = n_trees
        self.model = None
        self.environment_stats = {}
    
    def compute_algorithm_scores(self, df):
        #computes performance scores for each algorithm in each environment
        #this builds the training data for environment-level predictions
        
        results = []
        
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            
            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]
                
                #compute aggregate statistics
                stats = {
                    'environment': env,
                    'algorithm': algo,
                    'mean_reward': algo_data['mean_group_reward'].mean(),
                    'max_reward': algo_data['mean_group_reward'].max(),
                    'reward_std': algo_data['mean_group_reward'].std(),
                    'final_reward': algo_data.iloc[-1]['mean_group_reward'] if len(algo_data) > 0 else 0,
                    'avg_memory': algo_data['memory_usage_avg_mb'].mean(),
                    'avg_cpu': algo_data['cpu_usage_avg_mb'].mean(),
                    'total_steps': algo_data['step'].max(),
                    'n_samples': len(algo_data)
                }
                results.append(stats)
        
        return pd.DataFrame(results)
    
    def find_best_algorithm(self, scores_df):
        #for each environment, finds the best performing algorithm
        #uses final_reward as the main metric (can be changed)
        
        best = scores_df.loc[scores_df.groupby('environment')['final_reward'].idxmax()]
        return best[['environment', 'algorithm', 'final_reward']]
    
    def prepare_recommendation_data(self, df):
        #prepares data for training the recommendation model
        #features are early training metrics, label is best algorithm
        
        #get algorithm scores
        scores = self.compute_algorithm_scores(df)
        
        #find best algorithm per environment
        best = self.find_best_algorithm(scores)
        
        #store for reference
        self.environment_stats = scores
        
        return scores, best
    
    def recommend_algorithm(self, environment_name, early_metrics):
        #given an environment and early training metrics, recommends best algorithm
        #this is the main prediction function to use
        
        #TODO: implement after we have more training data from different environments
        #for now just return based on stored stats
        
        if len(self.environment_stats) == 0:
            print("Warning: No environment stats computed yet")
            return None
        
        env_data = self.environment_stats[self.environment_stats['environment'] == environment_name]
        
        if len(env_data) == 0:
            print(f"Warning: No data for environment {environment_name}")
            return None
        
        #return algorithm with highest final reward
        best_idx = env_data['final_reward'].idxmax()
        return env_data.loc[best_idx, 'algorithm']


def compare_algorithms(df):
    #quick comparison of algorithms in the dataset
    #prints summary statistics
    
    print("\n=== Algorithm Comparison ===\n")
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        print(f"Algorithm: {algo.upper()}")
        print(f"  Total samples: {len(algo_data)}")
        print(f"  Mean reward: {algo_data['mean_group_reward'].mean():.4f}")
        print(f"  Max reward: {algo_data['mean_group_reward'].max():.4f}")
        print(f"  Reward std: {algo_data['mean_group_reward'].std():.4f}")
        print(f"  Avg memory (MB): {algo_data['memory_usage_avg_mb'].mean():.1f}")
        print(f"  Avg CPU (%): {algo_data['cpu_usage_avg_mb'].mean():.1f}")
        print()


def predict_convergence(df, algorithm, threshold=0.1, window=10):
    #predicts if training will converge based on early metrics
    #looks at reward trend in early training
    
    algo_data = df[df['algorithm'] == algorithm].copy()
    
    if len(algo_data) < window:
        return None, "Not enough data"
    
    #compute rolling average of rewards
    algo_data['reward_ma'] = algo_data['mean_group_reward'].rolling(window=window).mean()
    
    #check trend in early training (first 20% of steps)
    early_cutoff = int(len(algo_data) * 0.2)
    early_data = algo_data.iloc[:early_cutoff]
    
    if len(early_data) < 2:
        return None, "Not enough early data"
    
    #simple linear regression to check trend
    x = np.arange(len(early_data))
    y = early_data['mean_group_reward'].values
    
    #fit line
    slope = np.polyfit(x, y, 1)[0]
    
    if slope > threshold:
        return True, f"Positive trend (slope={slope:.4f}), likely to converge"
    elif slope < -threshold:
        return False, f"Negative trend (slope={slope:.4f}), may not converge"
    else:
        return None, f"Flat trend (slope={slope:.4f}), uncertain"
