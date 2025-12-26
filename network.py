from layers.base_layer import Layer
from Losses.base_loss import Loss
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

    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params().items():
                yield param, layer.grads()[name]
