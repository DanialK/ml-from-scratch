import numpy as np

from supervised_learning.decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    return int(np.argmax(np.bincount(y.astype('int'))))


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_random_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_random_features = n_random_features
        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_random_features)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
