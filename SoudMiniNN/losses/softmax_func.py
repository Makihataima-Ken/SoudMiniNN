import numpy as np
from .base_loss import Loss

class SoftmaxCrossEntropy(Loss):
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        
        if y_true.ndim == 2:
            self.y_true = np.argmax(y_true, axis=1)
        else:
            self.y_true = y_true

        log_likelihood = -np.log(self.probs[range(len(self.y_true)), self.y_true] + 1e-7)
        return np.mean(log_likelihood)

    def backward(self):
        grad = self.probs.copy()
        
        y_true = self.y_true
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        grad[range(len(y_true)), y_true] -= 1
        return grad / len(y_true)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)