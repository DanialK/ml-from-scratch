import numpy as np

from base import BaseEstimator


def get_batch(X, y, batch_size):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        yield X[begin:end], y[begin:end]


class Model(BaseEstimator):
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        self._layers = []

    def fit(self, X_train, y_train, n_epochs, batch_size):
        for epoch in range(n_epochs):
            for X_batch, y_batch in get_batch(X_train, y_train, batch_size):
                y_batch = np.expand_dims(y_batch, axis=1)
                y_pred = self._forward_pass(X_batch, is_training=True)
                # loss = self.loss(y_batch, y_pred)
                loss_grad = self.loss.gradient(y_batch, y_pred)
                self._backward_pass(loss_grad)

    def predict(self, X):
        y_pred = self._forward_pass(X, is_training=False)
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        return y_pred

    def add(self, layer):

        if len(self._layers) != 0:
            layer.input_shape = self._layers[-1].output_shape

        self._layers.append(layer)

        layer.initialize(optimizer=self.optimizer)

    def set_is_trainable(self, is_trainable):
        for layer in self._layers:
            layer.is_trainable = is_trainable

    def _forward_pass(self, X, is_training):
        layer_output = X
        for layer in self._layers:
            layer_output = layer.forward_pass(layer_output, is_training)
        return layer_output

    def _backward_pass(self, loss_grad):
        layers_reversed = self._layers[::-1]
        for layer in layers_reversed:
            loss_grad = layer.backward_pass(loss_grad)
        return loss_grad
