import numpy as np
from soudmininn.layers.conv2d import Conv2D
from soudmininn.utils.grad_check import grad_check_module

def main():
    np.random.seed(0)
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)
    conv = Conv2D(3, 4, kernel_size=3, stride=1, padding=1)

    report = grad_check_module(conv, x, eps=1e-5, num_checks_per_param=20, seed=0)
    print("Conv2D grad-check max_relative_error:", report["max_relative_error"])

if __name__ == "__main__":
    main()
