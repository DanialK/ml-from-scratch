import copy
from abc import ABC, abstractmethod
import numpy as np

from supervised_learning.decision_tree import DecisionTreeRegressor


class LogisticLoss:
    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self._sigmoid(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, y, y_pred):
        p = self._sigmoid(y_pred)
        return -(y - p)

    def hess(self, y, y_pred):
        p = self._sigmoid(y_pred)
        return p * (1 - p)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


class SquareLoss:
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

    def hess(self, y, _):
        return np.ones(y.shape[0])


# DecisionTree used by Gradient Boosting algorithm to fit the residuals
class BoostedDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, loss, min_samples_split=2, max_depth=100, n_random_features=None):
        super(BoostedDecisionTreeRegressor, self).__init__(min_samples_split, max_depth, n_random_features)
        self.loss = loss

    @staticmethod
    def _split_y_and_y_pred(y_and_y_pred):
        col = int(y_and_y_pred.shape[1] / 2)
        y = y_and_y_pred[:, :col]
        y_pred = y_and_y_pred[:, col:]
        return y, y_pred

    def _impurity_calculation(self, y_and_y_pred, left_idx, right_idx, weights=None):
        y, y_pred = self._split_y_and_y_pred(y_and_y_pred)
        residuals = -self.loss.gradient(y, y_pred)
        # fit on residuals, meaning calculate gains on residuals, the base DecisionTreeRegressor uses variance reduction
        return super(BoostedDecisionTreeRegressor, self)._impurity_calculation(residuals, left_idx, right_idx)

    def _vote(self, y_and_y_pred):
        y, y_pred = self._split_y_and_y_pred(y_and_y_pred)
        # taylor approximation of optimal gamma (from StatQuest)
        numerator = np.sum(-self.loss.gradient(y, y_pred))
        denominator = np.sum(self.loss.hess(y, y_pred))
        # analytical solution of optimal gamma, basically results in the same formula
        # numerator = np.sum(y - y_pred)
        # denominator = y.shape[0]
        value = 0 if denominator == 0 else numerator / denominator
        return value

    def predict(self, X):
        y_pred = super(BoostedDecisionTreeRegressor, self).predict(X)
        return np.expand_dims(y_pred, axis=1)


class GradientBoosting(ABC):
    def __init__(self, tree, n_estimators, min_samples_split, max_depth, learning_rate):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self._initial_pred = None
        self._trees = [
            copy.copy(tree)
            for _ in range(self.n_estimators)
        ]

    def fit(self, X, y):
        self._initial_pred = self._initial_prediction(y)
        y = np.expand_dims(y, axis=1)
        y_pred = np.repeat(self._initial_pred, y.shape[0]).reshape(y.shape)

        for tree in self._trees:
            y_and_y_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_y_pred)
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred

    def _predict(self, X):
        y_pred = np.repeat(self._initial_pred, X.shape[0])
        y_pred = np.expand_dims(y_pred, axis=1)
        for tree in self._trees:
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred
        return y_pred.flatten()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    @abstractmethod
    def _initial_prediction(self, y):
        raise NotImplementedError()


# details reference: https://youtu.be/2xudPOBz-vs
class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators, min_samples_split, max_depth, learning_rate):
        loss = SquareLoss()
        tree = BoostedDecisionTreeRegressor(loss, min_samples_split, max_depth)
        super(GradientBoostingRegressor, self).__init__(tree, n_estimators, min_samples_split, max_depth, learning_rate)

    def predict(self, X):
        return self._predict(X)

    def _initial_prediction(self, y):
        return np.mean(y)


# details reference: https://youtu.be/jxuNLH5dXCs
class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators, min_samples_split, max_depth, learning_rate):
        loss = LogisticLoss()
        tree = BoostedDecisionTreeRegressor(loss, min_samples_split, max_depth)
        super(GradientBoostingClassifier, self).__init__(tree, n_estimators, min_samples_split, max_depth, learning_rate)

    def predict(self, X):
        p = self._log_odds_to_prob(self._predict(X))
        y_pred = np.array(p > 0.5, dtype=int)
        return y_pred

    @staticmethod
    def _log_odds_to_prob(log_odds_pred):
        return np.exp(log_odds_pred + 1e-15) / (1 + np.exp(log_odds_pred + 1e-15))

    def _initial_prediction(self, y):
        pos = np.sum(y)
        neg = y.shape[0] - pos
        odds = 0 if neg == 0 else pos / neg
        return np.log(odds)
