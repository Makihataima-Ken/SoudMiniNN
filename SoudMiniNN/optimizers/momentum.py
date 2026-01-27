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

def state_dict(self) -> dict:
    v_list = [self.v.get(id(p), np.zeros_like(p.data)) for p in self.params]
    return {
        "type":"Momentum",
        "lr": self.lr,
        "momentum": self.momentum,
        "weight_decay": self.weight_decay,
        "v": [v.copy() for v in v_list],
    }

def load_state_dict(self, state: dict) -> None:
    self.lr = float(state.get("lr", self.lr))
    self.momentum = float(state.get("momentum", self.momentum))
    self.weight_decay = float(state.get("weight_decay", self.weight_decay))
    v_list = state.get("v", None)
    if isinstance(v_list, list) and len(v_list) == len(self.params):
        self.v = {id(p): v_list[i].copy() for i, p in enumerate(self.params)}
