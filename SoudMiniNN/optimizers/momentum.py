import numpy as np # type: ignore
from .base_optimizer import Optimizer

class Momentum(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}

    def step(self) -> None:
        for p in self.params:
            if not p.requires_grad:
                continue
            key = id(p)
            if key not in self.v:
                self.v[key] = np.zeros_like(p.data)
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            self.v[key] = self.momentum * self.v[key] - self.lr * grad
            p.data[...] = p.data + self.v[key]
