from .base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params:dict, grads:dict):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
