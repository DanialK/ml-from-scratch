import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=4000, regularization=None, C=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.C = C

    def _init_parameters(self, X):
        _, n_features = X.shape
        self.theta = np.zeros(n_features + 1)

    def fit(self, X, y):
        n_samples, _ = X.shape

        self._init_parameters(X)

        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)

        for i in range(self.n_iterations):
            y_pred = A.dot(self.theta)
            error = y_pred - y
            if self.regularization is None:
                self.theta -= self.learning_rate * (A.T @ error) / n_samples
            elif self.regularization == 'l2':
                self.theta -= self.learning_rate * (self.C * A.T @ error + + np.sum(self.theta)) / n_samples

    def predict(self, X):
        return X @ self.theta[1:] + self.theta[0]


class LinearRegressionV2:
    def __init__(self, regularization=None, lambda0=0.1):
        self.regularization = regularization
        self.lambda0 = lambda0

    def fit(self, X, y):
        n_samples, _ = X.shape
        A = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        if self.regularization is None:
            self.theta = np.linalg.pinv(A.T @ A) @ A.T @ y
        elif self.regularization == 'l2':
            self.theta = np.linalg.pinv(A.T @ A + self.lambda0) @ A.T @ y


    def predict(self, X):
        return X @ self.theta[1:] + self.theta[0]

