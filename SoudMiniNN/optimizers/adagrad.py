import numpy as np
from .base_optimizer import Optimizer

class AdaGrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.eps = eps
        self.weight_decay = weight_decay
        self.h = {}

    def step(self) -> None:
        for p in self.params:
            if not p.requires_grad:
                continue
            key = id(p)
            if key not in self.h:
                self.h[key] = np.zeros_like(p.data)
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            self.h[key] += grad * grad
            p.data[...] = p.data - self.lr * grad / (np.sqrt(self.h[key]) + self.eps)
