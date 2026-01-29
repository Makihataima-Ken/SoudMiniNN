from __future__ import annotations
from dataclasses import dataclass
import numpy as np # type: ignore
from typing import Optional

@dataclass
class Parameter:
    data: np.ndarray  # W = wight
    grad: Optional[np.ndarray] = None # for store dW (change) after backward
    requires_grad: bool = True # for check if the nerun static or need training
    name: str = "" # name of op(optinal)

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float64) # to convert numpy array 
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None: 
        if self.requires_grad and self.grad is not None:
            self.grad[...] = 0.0
