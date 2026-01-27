import numpy as np # type: ignore

import os
import sys

# Allow running the example directly from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.layers.pooling import MaxPool2D
from SoudMiniNN.utils.grad_check import grad_check_input

def main():
    np.random.seed(0)
    x = np.random.randn(2, 2, 4, 4).astype(np.float32)
    pool = MaxPool2D(kernel_size=2, stride=2)
    report = grad_check_input(pool, x, eps=1e-5, num_checks=30, seed=0)
    print("MaxPool2D input-grad-check max_relative_error:", report["max_relative_error"])

if __name__ == "__main__":
    main()
