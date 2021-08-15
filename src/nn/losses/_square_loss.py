import numpy as np

from nn.losses._loss import Loss


class SquareLoss(Loss):
    def loss(self, y, y_pred):
        return np.power((y - y_pred), 2) / 2

    def gradient(self, y, y_pred):
        return -(y - y_pred)
