import numpy as np
from collections import Counter


def entropy1(y):
    n_samples = y.shape[0]
    counts = Counter(y)
    px = np.array(list(counts.values())) / n_samples
    return -px.dot(np.log(px))


def entropy2(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps])


def entropy3(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -ps.dot(np.log2(ps))
