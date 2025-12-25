import numpy as np

class MeanSquaredError:
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)

    def backward(self):
        return 2 * self.diff / len(self.diff)