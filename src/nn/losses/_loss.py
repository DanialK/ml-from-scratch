from abc import ABC, abstractmethod


class Loss(ABC):
    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    @abstractmethod
    def loss(self, y, y_pred):
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, y, y_pred):
        raise NotImplementedError()
