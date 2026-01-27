import numpy as np
from ..core.module import Module
from ..core.parameter import Parameter

class BatchNorm1d(Module):
    """
    Educational BatchNorm for (N, D) tensors.
    Similar behavior to PyTorch BatchNorm1d for 2D inputs.

    Notes:
    - running_mean/var updated with momentum
    - uses biased variance (np.var with ddof=0)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gamma = Parameter(np.ones((1, num_features), dtype=np.float32), name="gamma")
        self.beta = Parameter(np.zeros((1, num_features), dtype=np.float32), name="beta")
        self.eps = eps
        self.momentum = momentum

        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

        # cache for backward
        self.x_centered = None
        self.std_inv = None
        self.x_hat = None
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        if self.training:
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)

            self.x_centered = x - self.batch_mean
            std = np.sqrt(self.batch_var + self.eps)
            self.std_inv = 1.0 / std
            self.x_hat = self.x_centered * self.std_inv

            # update running stats (PyTorch-like: running = (1-mom)*running + mom*batch)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * self.batch_var
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        # grad is dL/dy (y = gamma*x_hat + beta)
        N = grad.shape[0]
        self.gamma.grad[...] = np.sum(grad * self.x_hat, axis=0, keepdims=True)
        self.beta.grad[...] = np.sum(grad, axis=0, keepdims=True)

        dxhat = grad * self.gamma.data

        if not self.training:
            # for eval-mode, treat normalization constants as fixed
            return dxhat / np.sqrt(self.running_var + self.eps)

        # Backprop through normalization:
        # x_hat = (x - mean) / sqrt(var + eps)
        # Standard derivation
        dvar = np.sum(dxhat * self.x_centered * -0.5 * (self.batch_var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dxhat * -self.std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)

        dx = dxhat * self.std_inv + dvar * 2.0 * self.x_centered / N + dmean / N
        return dx
