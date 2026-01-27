import numpy as np
from soudmininn.core.sequential import Sequential
from soudmininn.layers.dense import Dense
from soudmininn.layers.activations import ReLU
from soudmininn.utils.grad_check import grad_check_module

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
