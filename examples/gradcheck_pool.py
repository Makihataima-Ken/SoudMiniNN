import os
import sys
import numpy as np # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.layers.pooling import MaxPool2D
from SoudMiniNN.utils.grad_check import grad_check_input


def main():
    np.random.seed(1)

    pool = MaxPool2D(kernel_size=2, stride=2)

    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    dout = np.random.randn(1, 1, 2, 2).astype(np.float32)

    input_report = grad_check_input(pool, x, dout=dout, eps=1e-4, num_checks=10)

    print("MaxPool2D input grad max relative error:", input_report["max_relative_error"])


if __name__ == "__main__":
    main()
