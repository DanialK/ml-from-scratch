import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples = X.shape[0]
        self.X_mean = np.mean(X, axis=0)
        X_centered = X - self.X_mean

        C = X_centered.T @ X_centered / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eig(C)

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.X_mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCAv2:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        n_samples = X.shape[0]
        self.X_mean = np.mean(X, axis=0)
        X_centered = X - self.X_mean

        C = X_centered.T @ X_centered / (n_samples - 1)

        U, S, V = np.linalg.svd(C)

        # self.U = U[:, :self.n_components]
        # self.S = S[:self.n_components]
        self.components = V[:self.n_components]

    def transform(self, X):
        X_centered = X - self.X_mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
