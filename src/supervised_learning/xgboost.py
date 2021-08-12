import numpy as np
from supervised_learning.decision_tree import DecisionTreeRegressor
from supervised_learning.gradient_boosting import GradientBoosting, SquareLoss, LogisticLoss


# details reference:
# - https://youtu.be/ZVFeW798-2I
# - https://arxiv.org/pdf/1603.02754.pdf

class XGBoostedDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, loss, reg_lambda, min_samples_split, max_depth, n_random_features=None):
        super(XGBoostedDecisionTreeRegressor, self).__init__(min_samples_split, max_depth, n_random_features)
        self.loss = loss
        self.reg_lambda = reg_lambda

    @staticmethod
    def _split_y_and_y_pred(y_and_y_pred):
        col = int(y_and_y_pred.shape[1] / 2)
        y = y_and_y_pred[:, :col]
        y_pred = y_and_y_pred[:, col:]
        return y, y_pred

    def _similarity(self, y, y_pred):
        numerator = np.power(np.sum(-self.loss.gradient(y, y_pred)), 2)
        denominator = np.sum(self.loss.hess(y, y_pred)) + self.reg_lambda
        return 0 if denominator == 0 else numerator / denominator

    def _impurity_calculation(self, y_and_y_pred, left_idx, right_idx, weights=None):
        y, y_pred = self._split_y_and_y_pred(y_and_y_pred)
        total_sim = self._similarity(y, y_pred)
        left_sim = self._similarity(y[left_idx], y_pred[left_idx])
        right_sim = self._similarity(y[right_idx], y_pred[right_idx])
        sim_reduction = -(total_sim - (len(left_idx) / len(y) * left_sim + len(right_idx) / len(y) * right_sim))

        return np.sum(sim_reduction)

    def _vote(self, y_and_y_pred):
        y, y_pred = self._split_y_and_y_pred(y_and_y_pred)
        numerator = np.sum(-self.loss.gradient(y, y_pred))
        denominator = np.sum(self.loss.hess(y, y_pred)) + self.reg_lambda
        return 0 if denominator == 0 else numerator / denominator

    def predict(self, X):
        y_pred = super(XGBoostedDecisionTreeRegressor, self).predict(X)
        return np.expand_dims(y_pred, axis=1)


class XGBoostRegressor(GradientBoosting):
    def __init__(self, reg_lambda, n_estimators, min_samples_split, max_depth, learning_rate):
        loss = SquareLoss()
        tree = XGBoostedDecisionTreeRegressor(loss, reg_lambda, min_samples_split, max_depth)
        super(XGBoostRegressor, self).__init__(tree, n_estimators, min_samples_split, max_depth, learning_rate)

    def predict(self, X):
        return self._predict(X)

    def _initial_prediction(self, _):
        return 0.5


class XGBoostClassifier(GradientBoosting):
    def __init__(self, reg_lambda, n_estimators, min_samples_split, max_depth, learning_rate):
        loss = LogisticLoss()
        tree = XGBoostedDecisionTreeRegressor(loss, reg_lambda, min_samples_split, max_depth)
        super(XGBoostClassifier, self).__init__(tree, n_estimators, min_samples_split, max_depth, learning_rate)

    def predict(self, X):
        p = self._log_odds_to_prob(self._predict(X))
        y_pred = np.array(p > 0.5, dtype=int)
        return y_pred

    @staticmethod
    def _log_odds_to_prob(log_odds_pred):
        return np.exp(log_odds_pred + 1e-15) / (1 + np.exp(log_odds_pred + 1e-15))

    def _initial_prediction(self, _):
        return 0.5
