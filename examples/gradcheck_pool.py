import numpy as np
from soudmininn.layers.pooling import MaxPool2D
from soudmininn.utils.grad_check import grad_check_input

def main():
    np.random.seed(0)
    x = np.random.randn(2, 2, 4, 4).astype(np.float32)
    pool = MaxPool2D(kernel_size=2, stride=2)
    report = grad_check_input(pool, x, eps=1e-5, num_checks=30, seed=0)
    print("MaxPool2D input-grad-check max_relative_error:", report["max_relative_error"])

if __name__ == "__main__":
    main()
