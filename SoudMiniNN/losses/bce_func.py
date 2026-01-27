import numpy as np # type: ignore
from .base_loss import Loss

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

class BCEWithLogitsLoss(Loss):
    """
    Binary Cross-Entropy with logits (stable), like torch.nn.BCEWithLogitsLoss.
    targets should be 0/1 floats with same shape as logits.
    """
    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction
        self._logits = None
        self._targets = None
        self._sig = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        logits = logits.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)

        self._logits = logits
        self._targets = targets

        # stable: max(x,0) - x*y + log(1+exp(-|x|))
        loss = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
        if self.reduction == "sum":
            return float(np.sum(loss))
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        if self._logits is None or self._targets is None:
            raise RuntimeError("BCEWithLogitsLoss.backward called before forward")

        sig = _sigmoid(self._logits)
        self._sig = sig
        grad = (sig - self._targets)
        if self.reduction == "mean":
            grad = grad / grad.size
        return grad
