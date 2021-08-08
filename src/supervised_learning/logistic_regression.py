import numpy as np
from base import BaseEstimator


class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

    def _init_parameters(self, X):
        _, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

    def fit(self, X, y):
        n_samples = X.shape[0]

        self._init_parameters(X)
        for i in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self._weights + self._bias)
            error = y_pred - y
            if self.regularization is None:
                self._weights -= self.learning_rate * X.T @ error / n_samples
                self._bias -= self.learning_rate * np.sum(error) / n_samples
            elif self.regularization == 'l2':
                self._weights -= self.learning_rate * (
                            self.C * (X.T @ error) + np.sum(self._weights) + self._bias) / n_samples
                self._bias -= self.learning_rate * (self.C * np.sum(error)) / n_samples

    def predict_proba(self, X):
        return self._sigmoid(X @ self._weights + self._bias)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return np.round(y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class LogisticRegressionV2(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

    def _init_parameters(self, X):
        _, n_features = X.shape
        self._theta = np.zeros(n_features + 1)

    def fit(self, X, y):
        n_samples = X.shape[0]

        self._init_parameters(X)

        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)

        for i in range(self.n_iterations):
            y_pred = self._sigmoid(A @ self._theta)
            error = y_pred - y
            if self.regularization is None:
                self._theta -= self.learning_rate * (A.T @ error) / n_samples
            elif self.regularization == 'l2':
                self._theta -= self.learning_rate * (self.C * (A.T @ error) + np.sum(self._theta)) / n_samples

    def predict(self, X):
        y_pred = self._sigmoid(X.dot(self._theta[1:]) + self._theta[0])
        return np.round(y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
