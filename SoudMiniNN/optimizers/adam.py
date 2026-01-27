import numpy as np
from .base_optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self) -> None:
        self.t += 1
        for p in self.params:
            if not p.requires_grad:
                continue
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p.data)
                self.v[key] = np.zeros_like(p.data)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            p.data[...] = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
