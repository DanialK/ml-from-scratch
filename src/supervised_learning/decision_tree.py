from typing import Union
import numpy as np
from dataclasses import dataclass

from base import BaseEstimator


def entropy(y, weights=None):
    hist = np.bincount(y, weights)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gini(y, weights=None):
    if weights is None:
        weights = np.full(y.shape[0], 1 / y.shape[0])
    hist = np.bincount(y, weights)
    ps = hist / np.sum(weights)
    return 1 - np.sum([p ** 2 for p in ps])

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


class DecisionTree(BaseEstimator):
    def __init__(self, min_samples_split=2, max_depth=100, n_random_features=None, criterion='entropy'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_random_features = n_random_features
        self.criterion = criterion

    def fit(self, X, y, weights=None):
        self.n_random_features = X.shape[1] if not self.n_random_features else min(self.n_random_features, X.shape[1])
        self._root = self._grow(X, y, weights)

    def predict(self, X):
        y_pred = [self._traverse(x, self._root) for x in X]
        return np.array(y_pred)

    def _grow(self, X, y, weights=None, depth=0):
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

        best_feature_idx, best_threshold = self._best_criteria(X, y, random_feature_idx, weights)
        best_feature = X[:, best_feature_idx]
        left_idx, right_idx = split(best_feature, best_threshold)

        return Node(
            best_feature_idx,
            best_threshold,
            self._grow(X[left_idx, :], y[left_idx], weights[left_idx] if weights is not None else None, depth + 1),
            self._grow(X[right_idx, :], y[right_idx], weights[right_idx] if weights is not None else None, depth + 1)
        )

    def _best_criteria(self, X, y, random_feature_idx, weights=None):
        best_feature_idx = None
        best_threshold = None
        best_grain = -1
        for feat_idx in random_feature_idx:
            feat_column = X[:, feat_idx]
            thresholds = np.unique(feat_column)
            for threshold in thresholds:
                gain = self._information_gain(feat_column, y, threshold, weights)
                if gain > best_grain:
                    best_feature_idx = feat_idx
                    best_threshold = threshold
                    best_grain = gain

        return best_feature_idx, best_threshold

    def _information_gain(self, feat_column, y, threshold, weights=None):
        left_idx, right_idx = split(feat_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        y_quality = self._split_quality(y, weights)
        left_y_quality = self._split_quality(y[left_idx], (weights[left_idx] if weights is not None else None))
        right_y_quality = self._split_quality(y[right_idx], (weights[right_idx] if weights is not None else None))
        y_split_quality = len(left_idx) / len(y) * left_y_quality + len(right_idx) / len(y) * right_y_quality

        return y_quality - y_split_quality

    def _split_quality(self, y, weights=None):
        if self.criterion == 'entropy':
            return entropy(y, weights)
        elif self.criterion == 'gini':
            return gini(y, weights)
        else:
            raise f"criterion={self.criterion} not supported"

    def _vote(self, y):
        return np.argmax(np.bincount(y.astype('int')))

    def _traverse(self, x, node):
        if isinstance(node, Leaf):
            return node.value

        feat_inx, threshold = node.feat, node.threshold

        if x[feat_inx] <= threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
