import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.y_true = y_true
        log_likelihood = -np.log(self.probs[range(len(y_true)), y_true])
        return np.mean(log_likelihood)

    def backward(self):
        grad = self.probs.copy()
        grad[range(len(self.y_true)), self.y_true] -= 1
        return grad / len(self.y_true)
