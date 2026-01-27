import os
import sys
import numpy as np # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.core.sequential import Sequential
from SoudMiniNN.layers.dense import Dense
from SoudMiniNN.layers.activations import ReLU
from SoudMiniNN.utils.grad_check import grad_check_module, grad_check_input


def main():
    np.random.seed(42)

    model = Sequential(
        Dense(5, 4, init="xavier"),
        ReLU(),
        Dense(4, 3, init="xavier"),
    )

    x = np.random.randn(2, 5).astype(np.float32)
    dout = np.random.randn(2, 3).astype(np.float32)

    param_report = grad_check_module(model, x, dout=dout, eps=1e-5, num_checks_per_param=8)
    input_report = grad_check_input(model, x, dout=dout, eps=1e-5, num_checks=10)

    print("Parameter grad check max relative error:", param_report["max_relative_error"])
    print("Input grad check max relative error:", input_report["max_relative_error"])


if __name__ == "__main__":
    main()
