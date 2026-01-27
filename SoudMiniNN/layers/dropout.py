import numpy as np # type: ignore
from ..core.module import Module

class Dropout(Module):
    def __init__(self, rate: float = 0.5, seed: int | None = None):
        super().__init__()
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.mask = None
        self.rng = np.random.default_rng(seed)

    def forward(self, x):
        if self.training:
            self.mask = (self.rng.random(x.shape) < self.keep_prob).astype(x.dtype)
            # inverted dropout: scale during training so eval is identity
            return x * self.mask / self.keep_prob
        self.mask = None
        return x

    def backward(self, grad):
        if self.training and self.mask is not None:
            return grad * self.mask / self.keep_prob
        return grad
