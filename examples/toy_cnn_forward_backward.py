import os
import sys
import numpy as np # type: ignore

# Allow running the example directly from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.core.sequential import Sequential
from SoudMiniNN.layers.conv2d import Conv2D
from SoudMiniNN.layers.pooling import MaxPool2D
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.layers.flatten import Flatten
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.losses.softmax_func import CrossEntropyLoss
from SoudMiniNN.optimizers.adam import Adam

def main():
    np.random.seed(0)

    # Fake image batch: N=4, C=1, H=8, W=8
    x = np.random.randn(4, 1, 8, 8).astype(np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int64)  # 3 classes

    model = Sequential(
        Conv2D(1, 4, kernel_size=3, stride=1, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        Dense(4 * 4 * 4, 3, init="xavier"),
    )

    loss_fn = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-2)

    # One training step
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)

    dlogits = loss_fn.backward()
    model.backward(dlogits)
    opt.step()
    model.zero_grad()

    print("loss:", loss)
    print("logits shape:", logits.shape)

if __name__ == "__main__":
    main()
