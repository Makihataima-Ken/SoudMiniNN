import numpy as np # type: ignore
from .network import Network
from .optimizers.base_optimizer import Optimizer

class Trainer:
    def __init__(self, model: Network, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y):
        self.model.train()
        logits = self.model.forward(x)
        loss = self.model.loss_fn.forward(logits, y)

        dlogits = self.model.loss_fn.backward()
        self.model.backward(dlogits)

        self.optimizer.step()
        self.model.zero_grad()

        return loss

    def fit(self, x, y, epochs: int = 100, batch_size: int = 32, seed: int | None = None, print_every: int = 10):
        rng = np.random.default_rng(seed)
        n = x.shape[0]

        for e in range(1, epochs + 1):
            perm = rng.permutation(n)
            x_shuffled = x[perm]
            y_shuffled = y[perm]

            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                xb = x_shuffled[i:i+batch_size]
                yb = y_shuffled[i:i+batch_size]
                loss = self.train_step(xb, yb)
                epoch_loss += loss * xb.shape[0]

            epoch_loss /= n
            if (e % print_every) == 0 or e == 1:
                print(f"Epoch {e:03d} | loss={epoch_loss:.6f}")

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))
