from .layers.base_layer import Layer
from .losses.base_loss import Loss
from typing import List
import numpy as np

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

    
    def predict(self, X):
        logits = self.forward(X, training=False)
        if logits.shape[1] == 1:
            return (logits > 0.5).astype(int)
        else:
            return np.argmax(logits, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def loss(self, X, y):
        logits = self.forward(X, training=False)
        return self.loss_func.forward(logits, y)
    
    def gradient(self, X, y):
        # Performs full forward + backward to compute gradients
        logits = self.forward(X, training=True)
        _ = self.loss_func.forward(logits, y)  # needed for backward state
        grad = self.loss_func.backward()
        self.backward(grad)
        return self.get_grads()
