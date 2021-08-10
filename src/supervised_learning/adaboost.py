import numpy as np
import copy

from supervised_learning.decision_tree import DecisionTreeClassifier


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.error = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        self.error = float('inf')

        for feature_index in range(n_features):
            X_col = X[:, feature_index]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                polarity = 1
                y_pred = np.ones(n_samples)
                y_pred[X_col < threshold] = -1
                error = np.sum(weights[y != y_pred])

                # if the error is larger than 50%, then this stump is a good enough weak classifier for the opposite
                # class, so we flip the polarity
                if error > 0.5:
                    error = 1 - error
                    polarity = -1

                if error < self.error:
                    self.polarity = polarity
                    self.feature_idx = feature_index
                    self.threshold = threshold
                    self.error = error

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


# https://medium.com/analytics-vidhya/implementing-an-adaboost-classifier-from-scratch-e30ef86e9f1b
class Adaboost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self._clfs = []

    def fit(self, X, y_orig, eps=1e-10):
        # Ensure labels are transformed form 1 vs 0 --> 1 vs -1
        y = copy.copy(y_orig)
        y[y == 0] = -1

        n_samples, n_features = X.shape

        weights = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_clf):
            clf = DecisionStump()
            clf.fit(X, y, weights)
            predictions = clf.predict(X)

            error = np.sum(weights[y != predictions])
            alpha = 0.5 * np.log((1.0 - error + eps) / (error + eps))

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self._clfs.append((alpha, clf))

    def predict(self, X):
        y_preds = np.array([alpha * clf.predict(X) for alpha, clf in self._clfs]).T
        y_pred = np.sign(np.sum(y_preds, axis=1))
        y_pred[y_pred == -1] = 0
        return np.array(y_pred, dtype=int)


class AdaboostV2(Adaboost):

    def fit(self, X, y_orig, eps=1e-10):
        # Ensure labels are transformed form 1 vs 0 --> 1 vs -1
        y = copy.copy(y_orig)
        y[y == 0] = -1

        n_samples, n_features = X.shape

        weights = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_clf):
            clf = DecisionTreeClassifier(max_depth=1, criterion='gini')
            clf.fit(X, np.clip(y, 0, 1), weights)
            predictions = clf.predict(X)
            predictions[predictions == 0] = -1

            error = np.sum(weights[y != predictions])
            alpha = 0.5 * np.log((1.0 - error + eps) / (error + eps))

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            self._clfs.append((alpha, clf))

    def predict(self, X):
        y_preds = np.array([alpha * np.where(clf.predict(X) == 0, -1, 1) for alpha, clf in self._clfs]).T
        y_pred = np.sign(np.sum(y_preds, axis=1))
        y_pred[y_pred == -1] = 0
        return np.array(y_pred, dtype=int)
