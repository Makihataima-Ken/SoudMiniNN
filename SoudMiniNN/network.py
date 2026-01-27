import numpy as np # type: ignore
from .core.module import Module
from .core.sequential import Sequential
from .losses.base_loss import Loss

class Network(Module):
    """
    Thin wrapper around Sequential to match your previous API names.
    """
    def __init__(self, layers: list[Module], loss: Loss):
        super().__init__()
        self.model = Sequential(*layers)
        self.loss_fn = loss

    def forward(self, x):
        self.model.training = self.training
        return self.model.forward(x)

    def backward(self, grad):
        return self.model.backward(grad)

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        self.model.zero_grad()

    def predict(self, X):
        self.eval()
        logits = self.forward(X)
        if logits.ndim == 2 and logits.shape[1] > 1:
            return np.argmax(logits, axis=1)
        return (logits > 0.5).astype(int)
