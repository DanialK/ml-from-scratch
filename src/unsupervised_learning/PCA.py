import numpy as np

from base import BaseEstimator


class PCA(BaseEstimator):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples = X.shape[0]
        self._X_mean = np.mean(X, axis=0)
        X_centered = X - self._X_mean

        C = X_centered.T @ X_centered / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eig(C)

        self._components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self._X_mean
        return X_centered @ self._components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCAv2(BaseEstimator):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples = X.shape[0]
        self._X_mean = np.mean(X, axis=0)
        X_centered = X - self._X_mean

        C = X_centered.T @ X_centered / (n_samples - 1)

        U, S, V = np.linalg.svd(C)

        # self.U = U[:, :self.n_components]
        # self.S = S[:self.n_components]
        self._components = V[:self.n_components]

    def transform(self, X):
        X_centered = X - self._X_mean
        return X_centered @ self._components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
