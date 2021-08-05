from abc import ABC, abstractmethod

import numpy as np


class BaseKNN(ABC):

    def __init__(self, k):
        self.k = k
        super().__init__()

    @abstractmethod
    def _vote(self, k_y):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=0))
            idx = np.argsort(dist)[:self.k]
            k_y = self.y_train[idx]
            y_pred.append(self._vote(k_y))
        return y_pred


class KNNClassifier(BaseKNN):

    def __init__(self, k):
        super().__init__(k)

    def _vote(self, k_y):
        return np.argmax(np.bincount(k_y.astype('int')))


class KNNRegressor(BaseKNN):

    def __init__(self, k):
        super().__init__(k)

    def _vote(self, k_y):
        return np.mean(k_y)
