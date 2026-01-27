import numpy as np # type: ignore

import os
import sys

# Allow running the example directly from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.core.sequential import Sequential
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.utils.grad_check import grad_check_module

def main():
    np.random.seed(0)
    x = np.random.randn(5, 4).astype(np.float32)

    model = Sequential(
        Dense(4, 6, init="xavier"),
        ReLU(),
        Dense(6, 3, init="xavier"),
    )

    report = grad_check_module(model, x, eps=1e-5, num_checks_per_param=20, seed=0)
    print("Dense/ReLU/Dense grad-check max_relative_error:", report["max_relative_error"])

if __name__ == "__main__":
    main()
