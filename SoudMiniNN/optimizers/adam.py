import numpy as np # type: ignore
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

def state_dict(self) -> dict:
    m_list = [self.m.get(id(p), np.zeros_like(p.data)) for p in self.params]
    v_list = [self.v.get(id(p), np.zeros_like(p.data)) for p in self.params]
    return {
        "type":"Adam",
        "lr": self.lr,
        "beta1": self.beta1,
        "beta2": self.beta2,
        "eps": self.eps,
        "weight_decay": self.weight_decay,
        "t": self.t,
        "m": [m.copy() for m in m_list],
        "v": [v.copy() for v in v_list],
    }

def load_state_dict(self, state: dict) -> None:
    self.lr = float(state.get("lr", self.lr))
    self.beta1 = float(state.get("beta1", self.beta1))
    self.beta2 = float(state.get("beta2", self.beta2))
    self.eps = float(state.get("eps", self.eps))
    self.weight_decay = float(state.get("weight_decay", self.weight_decay))
    self.t = int(state.get("t", self.t))
    m_list = state.get("m", None)
    v_list = state.get("v", None)
    if isinstance(m_list, list) and len(m_list) == len(self.params):
        self.m = {id(p): m_list[i].copy() for i, p in enumerate(self.params)}
    if isinstance(v_list, list) and len(v_list) == len(self.params):
        self.v = {id(p): v_list[i].copy() for i, p in enumerate(self.params)}
