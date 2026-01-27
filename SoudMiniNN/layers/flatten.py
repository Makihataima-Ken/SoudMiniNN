import numpy as np # type: ignore
from ..core.module import Module

class Flatten(Module):
    """Flatten (N, C, H, W) or (N, ...) into (N, D)."""
    def __init__(self):
        super().__init__()
        self._orig_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._orig_shape is None:
            raise RuntimeError("Flatten.backward called before forward")
        return dout.reshape(self._orig_shape)
