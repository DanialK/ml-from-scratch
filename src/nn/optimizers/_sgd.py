import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self._prev_update = None

    def update(self, w, grad):
        if self._prev_update is None:
            self._prev_update = np.zeros(w.shape)
        update = self.momentum * self._prev_update + (1-self.momentum) * grad
        self._prev_update = update
        return w - self.learning_rate * update