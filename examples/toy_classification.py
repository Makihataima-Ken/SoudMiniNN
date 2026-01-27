import numpy as np # type: ignore
import os
import sys

# Allow running the example directly from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from SoudMiniNN.network import Network
from SoudMiniNN.trainer import Trainer
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.losses.softmax_func import CrossEntropyLoss
from SoudMiniNN.optimizers.adam import Adam

def make_toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like but not exactly
    return X, y

X, y = make_toy()

model = Network(
    layers=[Dense(2, 16), ReLU(), Dense(16, 2)],
    loss=CrossEntropyLoss()
)
opt = Adam(model.parameters(), lr=1e-2)

trainer = Trainer(model, opt)
trainer.fit(X, y, epochs=100, batch_size=32, seed=0, print_every=20)
print("acc:", trainer.accuracy(X, y))
