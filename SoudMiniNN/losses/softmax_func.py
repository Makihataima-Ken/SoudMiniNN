import numpy as np
from .base_loss import Loss

def _log_softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    # log_softmax(x) = x - log(sum(exp(x)))
    x_shift = logits - np.max(logits, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x_shift), axis=axis, keepdims=True) + 1e-12)
    return x_shift - logsumexp

class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss over class logits.
    - logits: (N, C)
    - targets: (N,) class indices OR (N, C) one-hot
    Returns mean loss by default (PyTorch-like).
    """
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        self.log_probs = None
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        self.log_probs = _log_softmax(logits, axis=1)
        self.probs = np.exp(self.log_probs)

        if targets.ndim == 2:
            self.targets = np.argmax(targets, axis=1)
        else:
            self.targets = targets.astype(int)

        N = logits.shape[0]
        loss = -self.log_probs[np.arange(N), self.targets]
        if self.reduction == "mean":
            return float(np.mean(loss))
        return float(np.sum(loss))

    def backward(self):
        # dL/dlogits = (softmax - one_hot)/N for mean reduction
        N, C = self.probs.shape
        grad = self.probs.copy()
        grad[np.arange(N), self.targets] -= 1.0
        if self.reduction == "mean":
            grad /= N
        return grad
