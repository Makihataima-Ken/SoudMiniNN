import numpy as np
from .base_layer import Layer

class BatchNorm(Layer):
    def __init__(self, input_size, momentum=0.9):
        # FIX: Initialize gamma and beta as learnable parameters of the correct shape
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        self.momentum = momentum
        
        # These will be initialized on the first forward pass
        self.running_mean = None
        self.running_var = None

        # Placeholders for backward pass
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None
        self.dgamma = None
        self.dbeta = None

    # FIX: Corrected spelling of the 'training' argument
    def forward(self, x, training=True):
        
        if self.running_mean is None:
            # Initialize running_mean and running_var on the first pass
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])

        if training:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            # Store values needed for backward pass
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # Update running stats for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # Use running stats during inference
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out
    
    def backward(self, grad):
        # Calculate gradients for gamma and beta first
        self.dbeta = grad.sum(axis=0, keepdims=True)
        self.dgamma = np.sum(self.xn * grad, axis=0, keepdims=True)

        # Propagate gradient back to the input
        dxn = self.gamma * grad
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = -np.sum(dxc, axis=0)
        dx = dxc + dmu / self.batch_size
        
        return dx
    
    def params(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def grads(self):
        return {'gamma': self.dgamma, 'beta': self.dbeta}

