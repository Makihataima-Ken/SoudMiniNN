import numpy as np
from .base_loss import Loss

class SoftmaxCrossEntropy(Loss):
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        self.y_true = y_true
        log_likelihood = -np.log(self.probs[range(len(y_true)), y_true])
        return np.mean(log_likelihood)

    def backward(self):
        grad = self.probs.copy()
        grad[range(len(self.y_true)), self.y_true] -= 1
        return grad / len(self.y_true)
    
def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size