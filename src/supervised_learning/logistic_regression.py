import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _init_parameters(self, X):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        self._init_parameters(X)
        for i in range(self.n_iterations):
            y_pred = self._sigmoid(X.dot(self.weights) + self.bias)
            error = y_pred - y
            self.weights -= self.learning_rate * X.T @ error
            self.bias -= self.learning_rate * np.sum(error)

    def predict(self, X):
        y_pred = self._sigmoid(X.dot(self.weights) + self.bias)
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
