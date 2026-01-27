import numpy as np # type: ignore
from .network import Network
from .optimizers.base_optimizer import Optimizer
from .utils.nn_utils import clip_grad_norm_

class Trainer:
    def __init__(self, model: Network, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x, y, clip_grad_max_norm: float | None = None):
        self.model.train()
        logits = self.model.forward(x)
        loss = self.model.loss_fn.forward(logits, y)

        dlogits = self.model.loss_fn.backward()
        self.model.backward(dlogits)

        if clip_grad_max_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=float(clip_grad_max_norm))

        self.optimizer.step()
        self.model.zero_grad()

        return float(loss)

    def eval_step(self, x, y):
        self.model.eval()
        logits = self.model.forward(x)
        loss = self.model.loss_fn.forward(logits, y)
        return float(loss)

    def fit(
        self,
        x_train, y_train,
        x_val=None, y_val=None,
        epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        print_every: int = 10,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        restore_best: bool = True,
        clip_grad_max_norm: float | None = None,
    ):
        rng = np.random.default_rng(seed)
        n = x_train.shape[0]

        best_val = float("inf")
        best_state = None
        patience_left = early_stopping_patience

        for e in range(1, epochs + 1):
            if shuffle:
                perm = rng.permutation(n)
                x = x_train[perm]
                y = y_train[perm]
            else:
                x = x_train
                y = y_train

            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                xb = x[i:i+batch_size]
                yb = y[i:i+batch_size]
                loss = self.train_step(xb, yb, clip_grad_max_norm=clip_grad_max_norm)
                epoch_loss += loss * xb.shape[0]
            epoch_loss /= n

            val_loss = None
            if x_val is not None and y_val is not None:
                val_loss = self.eval_step(x_val, y_val)

                improved = (best_val - val_loss) > float(early_stopping_min_delta)
                if improved:
                    best_val = val_loss
                    if restore_best:
                        best_state = self.model.state_dict()
                    patience_left = early_stopping_patience
                else:
                    if early_stopping_patience is not None:
                        patience_left = patience_left - 1 if patience_left is not None else None

            if (e % print_every) == 0 or e == 1:
                if val_loss is None:
                    print(f"Epoch {e:03d} | loss={epoch_loss:.6f}")
                else:
                    print(f"Epoch {e:03d} | loss={epoch_loss:.6f} | val_loss={val_loss:.6f}")

            if early_stopping_patience is not None and patience_left is not None and patience_left <= 0:
                if restore_best and best_state is not None:
                    self.model.load_state_dict(best_state, strict=True)
                print(f"Early stopping at epoch {e:03d}. Best val_loss={best_val:.6f}")
                break

    def predict(self, X):
        return self.model.predict(X)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))
