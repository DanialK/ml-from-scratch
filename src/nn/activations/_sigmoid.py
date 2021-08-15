import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __call__(self, x):
        return sigmoid(x)

    def gradient(self, x):
        return sigmoid(x) * (1 - sigmoid(x))
