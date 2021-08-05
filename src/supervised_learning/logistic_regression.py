import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

    def _init_parameters(self, X):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        n_samples = X.shape[0]

        self._init_parameters(X)
        for i in range(self.n_iterations):
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            error = y_pred - y
            if self.regularization is None:
                self.weights -= self.learning_rate * X.T @ error / n_samples
                self.bias -= self.learning_rate * np.sum(error) / n_samples
            elif self.regularization == 'l2':
                self.weights -= self.learning_rate * (
                            self.C * (X.T @ error) + np.sum(self.weights) + self.bias) / n_samples
                self.bias -= self.learning_rate * (self.C * np.sum(error)) / n_samples

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return np.round(y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class LogisticRegressionV2:
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

    def _init_parameters(self, X):
        _, n_features = X.shape
        self.theta = np.zeros(n_features + 1)

    def fit(self, X, y):
        n_samples = X.shape[0]

        self._init_parameters(X)

        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)

        for i in range(self.n_iterations):
            y_pred = self._sigmoid(A @ self.theta)
            error = y_pred - y
            if self.regularization is None:
                self.theta -= self.learning_rate * (A.T @ error) / n_samples
            elif self.regularization == 'l2':
                self.theta -= self.learning_rate * (self.C * (A.T @ error) + np.sum(self.theta)) / n_samples

    def predict(self, X):
        y_pred = self._sigmoid(X.dot(self.theta[1:]) + self.theta[0])
        return np.round(y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
