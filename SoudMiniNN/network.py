from .layers.base_layer import Layer
from .losses.base_loss import Loss
from typing import List

class NeuralNetwork:
    def __init__(self, layers:List[Layer], loss:Loss):
        self.layers = layers
        self.loss_func = loss

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                params[f'layer_{i}_{name}'] = param
        return params

    def get_grads(self):
        grads = {}
        for i, layer in enumerate(self.layers):
            for name, grad in layer.grads().items():
                grads[f'layer_{i}_{name}'] = grad
        return grads
