import numpy as np

from nn.layers._layer import Layer


class Dropout(Layer):
    def __init__(self, rate, input_shape=None, seed=None):
        super().__init__(input_shape)
        self.rate = rate
        self.seed = seed

        self._layer_input = None
        self._mask = None

    def initialize(self, optimizer):
        pass

    def parameters(self):
        return 0

    def forward_pass(self, X, is_training):
        if is_training:
            self._mask = np.random.binomial(1, self.rate, size=X.shape) == 0
            return np.where(self._mask, X, 0) / (1 - self.rate)  # inverted dropout
        return X

    def backward_pass(self, grad):
        return np.where(self._mask, grad, 0)

    @property
    def output_shape(self):
        return self.input_shape
