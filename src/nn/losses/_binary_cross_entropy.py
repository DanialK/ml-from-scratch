import numpy as np

from nn.losses._loss import Loss


class BinaryCrossEntropy(Loss):
    def loss(self, y, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1-y) * np.log(1-p))

    def gradient(self, y, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y/p) + (1-y)/(1-p)
