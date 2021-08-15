import numpy as np


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.max(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
