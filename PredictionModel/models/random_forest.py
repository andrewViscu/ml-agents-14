#random forest implementation for ml-agents prediction
#manually implements forest with bootstrap sampling and feature subsets

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class MyRandomForest:
    #manually implements forest with random data sampling forcing tree to find different patterns in data
    #then has "democratic" voting between all trees
    
    def __init__(self, n_trees=50, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.feature_subsets = []
        self.classes_ = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        
        self.trees = []
        self.feature_subsets = []
        
        for i in range(self.n_trees):
            #bootstrap sampling - pick random samples with replacement
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            #random feature subset - sqrt of total features
            n_subset = max(1, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, n_subset, replace=False)
            X_subset = X_sample[:, feature_indices]
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=i)
            tree.fit(X_subset, y_sample)
            
            self.trees.append(tree)
            self.feature_subsets.append(feature_indices)
    
    def predict(self, X):
        all_predictions = []
        
        for tree, features in zip(self.trees, self.feature_subsets):
            X_subset = X[:, features]
            pred = tree.predict(X_subset)
            all_predictions.append(pred)
        
        #stack predictions and do majority voting
        all_predictions = np.array(all_predictions).T
        final_predictions = []
        
        for sample_preds in all_predictions:
            values, counts = np.unique(sample_preds, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        #returns probability for each class
        all_predictions = []
        
        for tree, features in zip(self.trees, self.feature_subsets):
            X_subset = X[:, features]
            pred = tree.predict(X_subset)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions).T
        probas = []
        
        for sample_preds in all_predictions:
            class_probs = []
            for c in self.classes_:
                prob = np.mean(sample_preds == c)
                class_probs.append(prob)
            probas.append(class_probs)
        
        return np.array(probas)


def cross_validation(X, y, k=5, model_class=None, **model_kwargs):
    #k-fold cross validation
    #splits data into k parts and tests on each part
    
    if model_class is None:
        model_class = MyRandomForest
    
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    fold_size = n // k
    cv_scores = []
    
    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        cv_scores.append(score)
        print(f"Fold {i + 1}: {score:.3f}")
    
    return np.mean(cv_scores), cv_scores


def analyze_feature_importance(forest, feature_names):
    #rough estimate of feature importance by counting how often each feature is used
    #not as accurate as sklearn but gives us an idea
    
    feature_counts = np.zeros(len(feature_names))
    
    for feature_indices in forest.feature_subsets:
        for idx in feature_indices:
            if idx < len(feature_names):
                feature_counts[idx] += 1
    
    #normalize
    if feature_counts.sum() > 0:
        feature_counts = feature_counts / feature_counts.sum()
    
    importance = dict(zip(feature_names, feature_counts))
    return importance
