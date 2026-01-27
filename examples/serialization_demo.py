import os
import sys
import pickle
import numpy as np # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.core.sequential import Sequential
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.losses.mse_func import MSELoss
from SoudMiniNN.optimizers.sgd import SGD


def main():
    np.random.seed(7)

    # simple regression model
    model = Sequential(Dense(3, 5, init="xavier"), ReLU(), Dense(5, 1, init="xavier"))
    loss_fn = MSELoss()
    opt = SGD(model.parameters(), lr=0.1)

    x = np.random.randn(4, 3).astype(np.float32)
    y = np.random.randn(4, 1).astype(np.float32)

    # one training step
    preds = model.forward(x)
    loss = loss_fn.forward(preds, y)
    dpreds = loss_fn.backward()
    model.backward(dpreds)
    opt.step()

    print("Loss after one step:", loss)

    # save
    state = model.state_dict()
    save_path = os.path.join(os.path.dirname(__file__), "demo_state.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(state, f)
    print(f"Saved state_dict to {save_path}")

    # load into a fresh model
    new_model = Sequential(Dense(3, 5, init="xavier"), ReLU(), Dense(5, 1, init="xavier"))
    with open(save_path, "rb") as f:
        loaded = pickle.load(f)
    new_model.load_state_dict(loaded, strict=True)

    # compare outputs
    old_out = model.forward(x)
    new_out = new_model.forward(x)
    diff = np.max(np.abs(old_out - new_out))
    print("Max difference between original and reloaded outputs:", diff)


if __name__ == "__main__":
    main()
