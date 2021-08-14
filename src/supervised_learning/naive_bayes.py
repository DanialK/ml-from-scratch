from abc import ABC
import numpy as np


def multivariate_normal(X, mean, cov):
    n_samples, n_features = X.shape
    cov_det = np.linalg.det(cov)
    if cov_det == 0.0:
        cov_det = np.finfo(float).eps
    cov_pinv = np.linalg.pinv(cov)
    normalising_coeff = np.power(2 * np.pi, n_features / 2) * np.sqrt(cov_det)
    X_centered = X - mean

    z = -0.5 * X_centered @ cov_pinv @ X_centered.T
    return np.exp(z).diagonal() / normalising_coeff


def normal(x, mean, var):
    z = -0.5 * np.power((x - mean), 2) / var
    normalising_coeff = np.sqrt(2 * np.pi * var)
    return np.exp(z) / normalising_coeff


def log_normal(x, mean, var):
    z = -0.5 * np.power((x - mean), 2) / var
    normalising_coeff = np.sqrt(2 * np.pi * var)
    return z - np.log(normalising_coeff)


class BaseGaussianNB(ABC):
    def __init__(self, var_smoothing):
        self.var_smoothing = var_smoothing

    def _init_parameters(self, X, y):
        y_counts = np.bincount(y)
        self._priors = y_counts / np.sum(y_counts)
        self._parameters = []
        self._epsilon = self.var_smoothing * np.var(X, axis=0).max()

    def fit(self, X, y):
        self._init_parameters(X, y)

        for label in range(self._priors.shape[0]):
            idx = y == label
            mean = np.mean(X[idx], axis=0)
            var = np.var(X[idx], axis=0)
            var += self._epsilon
            self._parameters.append((mean, var))


class GaussianNaiveBayesV1(BaseGaussianNB):
    def __init__(self, var_smoothing=1e-9):
        super().__init__(var_smoothing)

    def predict(self, X):
        y_prob = np.zeros((X.shape[0], len(self._priors)))
        for i, prior, (mean, var) in zip(range(len(self._priors)), self._priors, self._parameters):
            likelihoods = multivariate_normal(X, mean, np.diag(var))
            y_prob[:, i] = prior * likelihoods
        return y_prob.argmax(axis=1)


class GaussianNaiveBayesV2(BaseGaussianNB):
    def __init__(self, var_smoothing=1e-9):
        super().__init__(var_smoothing)

    def predict(self, X):
        n_samples, n_features = X.shape
        y_prob = np.zeros((n_samples, len(self._priors)))
        for label_idx, prior, (mean, var) in zip(range(len(self._priors)), self._priors, self._parameters):
            likelihoods = np.ones(n_samples)
            for feature_idx, f_mean, f_var in zip(range(n_features), mean, var):
                likelihoods *= normal(X[:, feature_idx], f_mean, f_var)
            y_prob[:, label_idx] = prior * likelihoods
        return y_prob.argmax(axis=1)


class GaussianNaiveBayesV3(BaseGaussianNB):
    def __init__(self, var_smoothing=1e-9):
        super().__init__(var_smoothing)

    def predict(self, X):
        y_prob = np.zeros((X.shape[0], len(self._priors)))
        for i, prior, (mean, var) in zip(range(len(self._priors)), self._priors, self._parameters):
            likelihoods = np.prod(normal(X, mean, var), axis=1)
            y_prob[:, i] = prior * likelihoods
        return y_prob.argmax(axis=1)


class GaussianNaiveBayesV4(BaseGaussianNB):
    def __init__(self, var_smoothing=1e-9):
        super().__init__(var_smoothing)

    def predict(self, X):
        y_prob = np.zeros((X.shape[0], len(self._priors)))
        for i, prior, (mean, var) in zip(range(len(self._priors)), self._priors, self._parameters):
            likelihoods = np.sum(log_normal(X, mean, var), axis=1)
            y_prob[:, i] = np.log(prior) + likelihoods
        return y_prob.argmax(axis=1)
