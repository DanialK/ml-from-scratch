import numpy as np

from base import BaseEstimator


def calculate_covariance_matrix(X):
    n_samples = X.shape[0]
    X_mean = X.mean(axis=0)
    X_centred = X - X_mean
    return 1 / (n_samples - 1) * X_centred.T @ X_centred


def multivariate_normal(X, mean, cov):
    n_samples, n_features = X.shape
    cov_det = np.linalg.det(cov)
    if cov_det == 0.0:
        cov_det = np.finfo(float).eps
    cov_pinv = np.linalg.pinv(cov)
    normalising_coeff = np.power(2 * np.pi, n_features / 2) * np.sqrt(cov_det)
    X_centered = X - mean

    return np.exp(-0.5 * X_centered @ cov_pinv @ X_centered.T).diagonal() / normalising_coeff


# reference: https://www.researchgate.net/publication/339456547/figure/fig14/AS:862099731406856@1582552001304/Pseudocode-of-the-expectation-maximization-EM-algorithm-for-Gaussian-mixture-modeling.ppm
class GMM(BaseEstimator):
    def __init__(self, n_components, max_iterations):
        self.n_components = n_components
        self.max_iterations = max_iterations

        self._clusters = None
        self._priors = None
        self._params = []
        self._responsibility = None

    def fit(self, X):
        self._init_parameters(X)

        for i in range(self.max_iterations):
            self._responsibility = self._expectation_step(X)
            self._maximisation_step(X)

        self._clusters = self.predict(X)

    def predict(self, X):
        # compute the responsibility using the latest parameters
        responsibility = self._expectation_step(X)
        return responsibility.argmax(axis=1)

    def _init_parameters(self, X):
        n_samples, n_features = X.shape
        self._priors = np.random.rand(self.n_components)
        self._priors = self._priors / self._priors.sum()

        for k in range(self.n_components):
            mean = X[np.random.choice(range(n_samples))]
            cov = calculate_covariance_matrix(X)
            self._params.append((mean, cov))

    def _expectation_step(self, X):
        n_samples = X.shape[0]
        weighted_likelihoods = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            prior = self._priors[k]
            (mean, cov) = self._params[k]
            likelihood = multivariate_normal(X, mean, cov)
            weighted_likelihoods[:, k] = likelihood * prior

        sum_weighted_likelihoods = weighted_likelihoods.sum(axis=1).reshape((n_samples, 1))
        responsibility = weighted_likelihoods / sum_weighted_likelihoods

        return responsibility

    def _maximisation_step(self, X):
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            responsibility_k = self._responsibility[:, k]
            sum_responsibility_k = np.sum(responsibility_k)
            mean = X.T @ responsibility_k / sum_responsibility_k
            X_centered = X - mean
            cov = X_centered.T @ np.diag(responsibility_k) @ X_centered / sum_responsibility_k
            prior = sum_responsibility_k / n_samples
            self._params[k] = (mean, cov)
            self._priors[k] = prior
