from __future__ import annotations
from dataclasses import dataclass
import numpy as np # type: ignore
from typing import Optional

@dataclass
class Parameter:
    """
    A trainable tensor-like object: holds data and gradient.

    Notes (educational):
    - We keep gradients explicitly (no autograd).
    - Optimizers update `data` in-place using `grad`.
    """
    data: np.ndarray
    grad: Optional[np.ndarray] = None
    requires_grad: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float32)
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        if self.requires_grad and self.grad is not None:
            self.grad[...] = 0.0
