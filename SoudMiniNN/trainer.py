import numpy as np
from .optimizers.base_optimizer import Optimizer
from .network import NeuralNetwork

class Trainer:
    def __init__(self, model:NeuralNetwork, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model.forward(x)
        loss = self.model.loss_func.forward(logits, y)
        grad = self.model.loss_func.backward()
        self.model.backward(grad)

        for layer in self.model.layers:
            if layer.params():
                self.optimizer.update(layer.params(), layer.grads())

        return loss

    def fit(self, x, y, epochs=100):
        for e in range(epochs):
            loss = self.train_step(x, y)
            if e % 10 == 0:
                print(f"Epoch {e}, Loss: {loss:.4f}")
                
    def accuracy(self, X, y):
        logits = self.model.forward(X, training=False)
        predictions = np.argmax(logits, axis=1)
        return np.mean(predictions == y)
