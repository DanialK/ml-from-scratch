import numpy as np


def get_batch(X, batch_size):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield X[begin:end]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample(X):
    return X > np.random.random_sample(size=X.shape)


# details reference:
# - https://youtu.be/wMb7cads0go
# - https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
# - 
class RBM:
    def __init__(self, n_hidden=128, k=1, learning_rate=0.1, batch_size=10, n_iterations=100):
        self.n_iterations = n_iterations
        self.k = k
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden

        self.weights = None
        self.h_bias = None
        self.v_bias = None

    def _initialize_parameters(self, X):
        n_features = X.shape[1]
        self.weights = np.random.uniform(size=(n_features, self.n_hidden))  # n_features x n_hidden
        self.h_bias = np.zeros(self.n_hidden)
        self.v_bias = np.zeros(n_features)

    def fit(self, X):
        self._initialize_parameters(X)

        for _ in range(self.n_iterations):
            for v0 in get_batch(X, self.batch_size):
                p_h_given_v0 = sigmoid(self.h_bias + v0 @ self.weights)  # n x n_hidden

                vk = v0
                p_h_given_vk = p_h_given_v0
                for _ in range(self.k):
                    hk = sample(p_h_given_vk)
                    p_v_given_hk = sigmoid(self.v_bias + hk @ self.weights.T)  # n x n_features
                    vk = sample(p_v_given_hk)
                    p_h_given_vk = sigmoid(self.h_bias + vk @ self.weights)  # n x n_hidden

                self.weights += self.learning_rate * (v0.T @ p_h_given_v0 - vk.T @ p_h_given_vk)
                self.h_bias += self.learning_rate * (p_h_given_v0.sum(axis=0) - p_h_given_vk.sum(axis=0))
                self.v_bias += self.learning_rate * (v0.sum(axis=0) - vk.sum(axis=0))

    def transform(self, X):
        p_h_given_v = sigmoid(self.h_bias + X @ self.weights)  # n x n_hidden
        hk = sample(p_h_given_v)
        p_v_given_hk = sigmoid(self.v_bias + hk @ self.weights.T)  # n x n_features
        return p_v_given_hk
