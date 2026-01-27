import numpy as np # type: ignore
from .base_loss import Loss

class MSELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = (y_pred - y_true) ** 2
        if self.reduction == "mean":
            return float(np.mean(loss))
        return float(np.sum(loss))

    def backward(self):
        # d/dy_pred ( (y_pred - y_true)^2 ) = 2*(y_pred - y_true)
        grad = 2.0 * (self.y_pred - self.y_true)
        if self.reduction == "mean":
            grad = grad / self.y_pred.size
        return grad
