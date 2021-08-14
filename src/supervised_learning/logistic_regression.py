import numpy as np
from base import BaseEstimator


class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

        self._weights = None
        self._bias = None

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
        y_prop = self.predict_proba(X)
        return np.round(y_prop)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


class LogisticRegressionV2(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

        self._theta = None

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

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self._theta[1:]) + self._theta[0])

    def predict(self, X):
        y_prop = self.predict_proba(X)
        return np.round(y_prop)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


class MultinomialLogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

        self._theta = None

    def _init_parameters(self, X, y):
        n_features = X.shape[1]
        n_labels = y.shape[1]
        self._theta = np.zeros((n_features + 1, n_labels))

    def fit(self, X, y):
        n_samples = X.shape[0]

        self._init_parameters(X, y)

        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)

        for i in range(self.n_iterations):
            y_pred = self._softmax(A @ self._theta)
            error = y_pred - y
            if self.regularization is None:
                self._theta -= self.learning_rate * (A.T @ error) / n_samples
            elif self.regularization == 'l2':
                self._theta -= self.learning_rate * (self.C * (A.T @ error) + np.sum(self._theta, axis=0)) / n_samples

    def predict(self, X):
        n_samples = X.shape[0]
        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        return self._softmax(A @ self._theta)

    @staticmethod
    def _softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=0)


class OneVsRestLogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

        self._models = []

    def fit(self, X, y):
        n_labels = y.shape[1]

        for label in range(n_labels):
            ovr_y = y.argmax(axis=1)
            idx = ovr_y == label
            ovr_y[idx] = 1
            ovr_y[np.invert(idx)] = 0
            model = LogisticRegression(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                regularization=self.regularization,
                C=self.C
            )
            model.fit(X, ovr_y)
            self._models.append(model)

    def predict(self, X):
        n_labels = len(self._models)
        n_samples = X.shape[0]

        y_pred = np.zeros((n_samples, n_labels))

        for i, model in enumerate(self._models):
            prob = model.predict_proba(X)
            y_pred[:, i] = prob

        return y_pred
