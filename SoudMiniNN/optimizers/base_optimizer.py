from abc import ABC, abstractmethod
from typing import Iterable, List
from ..core.parameter import Parameter

class Optimizer(ABC):
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params: List[Parameter] = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    @abstractmethod
    def step(self) -> None:
        pass

def state_dict(self) -> dict:
    """Return optimizer hyperparameters + internal state (educational)."""
    return {"lr": self.lr, "state": {}}

def load_state_dict(self, state: dict) -> None:
    """Load optimizer hyperparameters/state."""
    if "lr" in state:
        self.lr = float(state["lr"])
