from typing import Union

import numpy as np
from dataclasses import dataclass


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def split(x, threshold):
    left_idx = np.where(x <= threshold)[0]
    right_idx = np.where(x > threshold)[0]
    return left_idx, right_idx


@dataclass
class Node:
    feat: int
    threshold: np.float64
    left: Union['Node', 'Leaf']
    right: Union['Node', 'Leaf']


@dataclass
class Leaf:
    value: int


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_random_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_random_features = n_random_features

    def fit(self, X, y):
        self.n_random_features = X.shape[1] if not self.n_random_features else min(self.n_random_features, X.shape[1])
        self.root = self._grow(X, y)

    def predict(self, X):
        y_pred = [self._traverse(x, self.root) for x in X]
        return np.array(y_pred)

    def _grow(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
                n_samples < self.min_samples_split
                or depth > self.max_depth
                or n_labels == 1
        ):
            value = self._vote(y)
            return Leaf(value)

        random_feature_idx = np.random.choice(range(n_features), self.n_random_features, replace=False)

        best_feature_idx, best_threshold = self._best_criteria(X, y, random_feature_idx)
        best_feature = X[:, best_feature_idx]
        left_idx, right_idx = split(best_feature, best_threshold)

        return Node(
            best_feature_idx,
            best_threshold,
            self._grow(X[left_idx, :], y[left_idx], depth + 1),
            self._grow(X[right_idx, :], y[right_idx], depth + 1)
        )

    def _best_criteria(self, X, y, random_feature_idx):
        best_feature_idx = None
        best_threshold = None
        best_grain = -1
        for feat_idx in random_feature_idx:
            feat_column = X[:, feat_idx]
            thresholds = np.unique(feat_column)
            for threshold in thresholds:
                gain = self._information_gain(feat_column, y, threshold)
                if gain > best_grain:
                    best_feature_idx = feat_idx
                    best_threshold = threshold
                    best_grain = gain

        return best_feature_idx, best_threshold

    def _information_gain(self, feat_column, y, threshold):
        left_idx, right_idx = split(feat_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        y_entropy = entropy(y)
        left_y_entropy = entropy(y[left_idx])
        right_y_entropy = entropy(y[right_idx])
        y_split_entropy = len(left_idx) / len(y) * left_y_entropy + len(right_idx) / len(y) * right_y_entropy

        return y_entropy - y_split_entropy

    def _vote(self, y):
        return int(np.argmax(np.bincount(y.astype('int'))))

    def _traverse(self, x, node):
        if isinstance(node, Leaf):
            return node.value

        feat_inx, threshold = node.feat, node.threshold

        if x[feat_inx] <= threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
