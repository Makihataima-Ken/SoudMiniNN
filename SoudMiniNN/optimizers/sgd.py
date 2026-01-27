from .base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.params:
            if not p.requires_grad:
                continue
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            p.data[...] = p.data - self.lr * grad
