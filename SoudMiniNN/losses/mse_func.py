import numpy as np
from .base_loss import Loss

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)

    def backward(self):
        return 2 * self.diff / len(self.diff)