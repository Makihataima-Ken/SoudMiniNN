import numpy as np # type: ignore
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

def state_dict(self) -> dict:
    h_list = [self.h.get(id(p), np.zeros_like(p.data)) for p in self.params]
    return {
        "type":"AdaGrad",
        "lr": self.lr,
        "eps": self.eps,
        "weight_decay": self.weight_decay,
        "h": [h.copy() for h in h_list],
    }

def load_state_dict(self, state: dict) -> None:
    self.lr = float(state.get("lr", self.lr))
    self.eps = float(state.get("eps", self.eps))
    self.weight_decay = float(state.get("weight_decay", self.weight_decay))
    h_list = state.get("h", None)
    if isinstance(h_list, list) and len(h_list) == len(self.params):
        self.h = {id(p): h_list[i].copy() for i, p in enumerate(self.params)}
