import numpy as np
from .base_layer import Layer

class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.rand(*x.shape) > self.rate
            return x * self.mask
        else:
            return x * (1.0 - self.rate)

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask