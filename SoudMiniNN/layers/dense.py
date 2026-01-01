import numpy as np
from .base_layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, init ="he"):
        
        if init  == "he":
            self.W = self._he_init(input_size, output_size)
        elif init == "xavier" or init == "glorot":
            self.W = self._xavier_init(input_size, output_size)

        self.b = np.zeros((1, output_size))
        
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def params(self):
        return {'W': self.W, 'b': self.b}

    def grads(self):
        return {'W': self.dW, 'b': self.db}
    
    def _xavier_init(self, fan_in, fan_out):
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std
    
    def _he_init(self, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        return np.random.randn(fan_in, fan_out) * std

    
    
