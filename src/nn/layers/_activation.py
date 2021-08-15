from nn.activations import ReLU, Sigmoid
from nn.layers._layer import Layer

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
}


class Activation(Layer):
    def __init__(self, name, input_shape=None):
        super().__init__(input_shape)
        self.activation_name = name
        self.activation_func = activation_functions[name]()

        self._layer_input = None

    def initialize(self, optimizer):
        pass

    def parameters(self):
        return 0

    def forward_pass(self, X, is_training):
        self._layer_input = X
        return self.activation_func(X)

    def backward_pass(self, grad):
        return grad * self.activation_func.gradient(self._layer_input)

    @property
    def output_shape(self):
        return self.input_shape
