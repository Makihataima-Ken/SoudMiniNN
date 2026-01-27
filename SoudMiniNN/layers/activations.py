import numpy as np
from ..core.module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Softmax(Module):
    """
    Usually you DON'T need Softmax as a layer when using CrossEntropyLoss on logits.
    Keep it for inference/visualization.
    """
    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis
        self.out = None

    def forward(self, x):
        x_shift = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shift)
        self.out = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        return self.out

    def backward(self, grad):
        # general softmax backward is expensive; for training use CrossEntropyLoss(logits)
        # Here we implement a correct but slower version.
        y = self.out
        dx = np.empty_like(grad)
        for i in range(grad.shape[0]):
            yi = y[i].reshape(-1, 1)
            J = np.diagflat(yi) - yi @ yi.T
            dx[i] = (J @ grad[i].reshape(-1, 1)).reshape(-1)
        return dx
