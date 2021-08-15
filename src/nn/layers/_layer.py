from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, input_shape=None):
        self._input_shape = input_shape
        self._is_trainable = True

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape

    @property
    def is_trainable(self):
        return self._is_trainable

    @is_trainable.setter
    def is_trainable(self, is_trainable):
        self._is_trainable = is_trainable

    def layer_name(self):
        return self.__class__.__name__

    @abstractmethod
    def initialize(self, optimizer):
        raise NotImplementedError()

    @abstractmethod
    def parameters(self):
        raise NotImplementedError()

    @abstractmethod
    def forward_pass(self, X, is_training):
        raise NotImplementedError()

    @abstractmethod
    def backward_pass(self, grad):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()
