import numpy as np # type: ignore
from ..core.module import Module
from ..core.parameter import Parameter
from ..core.__init__ import he_init, xavier_init

class Dense(Module): # Linear  take many features and mixed there to make strong features
    def __init__(self, input_size: int, output_size: int, init: str = "he"):
        super().__init__() # to make the wight balanced
        if init.lower() in ("he", "kaiming"):
            W = he_init(input_size, output_size)
        elif init.lower() in ("xavier", "glorot"):
            W = xavier_init(input_size, output_size)
        else:
            raise ValueError(f"Unknown init='{init}' (use 'he' or 'xavier').")

        # convert the data to parametr
        self.W = Parameter(W, name="W")
        self.b = Parameter(np.zeros((1, output_size), dtype=np.float32), name="b") #broadcast
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad):
        # grad: dL/dout
        # dW = x^T @ grad, db = sum(grad)
        self.W.grad[...] = self.x.T @ grad
        self.b.grad[...] = np.sum(grad, axis=0, keepdims=True)
        # dx = grad @ W^T
        return grad @ self.W.data.T
