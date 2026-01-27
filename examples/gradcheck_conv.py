import os
import sys
import numpy as np # type: ignore

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SoudMiniNN.layers.conv2d import Conv2D
from SoudMiniNN.utils.grad_check import grad_check_module, grad_check_input


def main():
    np.random.seed(0)

    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)

    x = np.random.randn(2, 1, 4, 4).astype(np.float64)
    conv.W.data = conv.W.data.astype(np.float64)
    if conv.b is not None:
        conv.b.data = conv.b.data.astype(np.float64)
    
    param_report = grad_check_module(conv, x, eps=1e-5, num_checks_per_param=10)
    input_report = grad_check_input(conv, x, eps=1e-5, num_checks=20)



    print("Conv2D parameter grad max relative error:", param_report["max_relative_error"])
    print("Conv2D input grad max relative error:", input_report["max_relative_error"])


if __name__ == "__main__":
    main()
