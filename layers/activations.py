import numpy as np
from .base_layer import Layer

class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh(Layer):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)
