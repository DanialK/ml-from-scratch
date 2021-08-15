import copy

import numpy as np

from ._layer import Layer


class Dense(Layer):

    def __init__(self, n_units, input_shape=None):
        super().__init__(input_shape)
        self.n_units = n_units

        self._layer_input = None
        self._W = None
        self._w0 = None
        self._W_optimizer = None
        self._w0_optimizer = None

    def initialize(self, optimizer):
        limit = 1 / np.sqrt(self.input_shape[0])
        self._W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        # self._W = np.random.normal(size=(self.input_shape[0], self.n_units))
        self._w0 = np.zeros((1, self.n_units))
        self._W_optimizer = copy.copy(optimizer)
        self._w0_optimizer = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self._W.shape) + np.prod(self._w0.shape)

    def forward_pass(self, X, is_training):
        self._layer_input = X
        return X @ self._W + self._w0

    def backward_pass(self, grad):
        W_prev = self._W

        self._W = self._W_optimizer.update(W_prev, self._layer_input.T @ grad)
        self._w0 = self._w0_optimizer.update(self._w0, np.sum(grad, axis=0, keepdims=True))

        return grad @ W_prev.T

    @property
    def output_shape(self):
        return self.n_units,
