import numpy as np # type: ignore

import os
import sys

# Allow running the example directly from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle

from SoudMiniNN.network import Network
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.losses.softmax_func import CrossEntropyLoss
from SoudMiniNN.optimizers.adam import Adam

def main():
    np.random.seed(0)

    x = np.random.randn(10, 4).astype(np.float32)
    y = np.random.randint(0, 3, size=(10,)).astype(np.int64)

    model = Network(
        layers=[Dense(4, 8), ReLU(), Dense(8, 3)],
        loss=CrossEntropyLoss()
    )

    opt = Adam(model.parameters(), lr=1e-2)

    # one step
    model.train()
    logits = model.forward(x)
    loss = model.loss_fn.forward(logits, y)
    dlogits = model.loss_fn.backward()
    model.backward(dlogits)
    opt.step()
    model.zero_grad()

    sd = model.state_dict()

    with open("model_state.pkl", "wb") as f:
        pickle.dump(sd, f)

    # load into a new model
    model2 = Network(
        layers=[Dense(4, 8), ReLU(), Dense(8, 3)],
        loss=CrossEntropyLoss()
    )
    with open("model_state.pkl", "rb") as f:
        sd2 = pickle.load(f)
    model2.load_state_dict(sd2)

    # check same output
    out1 = model.forward(x)
    out2 = model2.forward(x)
    print("max abs diff after load:", float(np.max(np.abs(out1 - out2))))

if __name__ == "__main__":
    main()
