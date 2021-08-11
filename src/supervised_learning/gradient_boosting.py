import copy
from abc import ABC, abstractmethod
import numpy as np

from supervised_learning.decision_tree import DecisionTreeRegressor


def squared_loss_grad(y, y_pred):
    return -(y - y_pred)


def binary_cross_entropy_loss_grad(y, y_pred):
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)


class GradientBoosting(ABC):
    def __init__(self, tree, n_estimators, min_samples_split, max_depth, learning_rate):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self._trees = [
            copy.copy(tree)
            for _ in range(self.n_estimators)
        ]

    def fit(self, X, y):
        self._y_mean = np.mean(y)
        y_pred = np.repeat(self._y_mean, X.shape[0])

        for tree in self._trees:
            grad = self._loss_grad(y, y_pred)
            residuals = -grad
            tree.fit(X, residuals)
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred

    def _predict(self, X):
        y_pred = np.repeat(self._y_mean, X.shape[0])
        for tree in self._trees:
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred
        return y_pred

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def _loss_grad(self, y, y_pred):
        raise NotImplementedError()


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators, min_samples_split, max_depth, learning_rate):
        tree = DecisionTreeRegressor(min_samples_split, max_depth)
        super(GradientBoostingRegressor, self).__init__(tree, n_estimators, min_samples_split, max_depth, learning_rate)

    def _loss_grad(self, y, y_pred):
        return squared_loss_grad(y, y_pred)

    def predict(self, X):
        return self._predict(X)
