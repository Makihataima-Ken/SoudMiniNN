from __future__ import annotations
from typing import Iterable, List
from .module import Module

class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules_list: List[Module] = list(modules)

    def forward(self, x):
        for m in self.modules_list:
            # allow modules to read self.training if they want
            m.training = self.training
            x = m.forward(x)
        return x

    def backward(self, grad):
        for m in reversed(self.modules_list):
            grad = m.backward(grad)
        return grad

    def named_modules(self):
        for i, m in enumerate(self.modules_list):
            yield (str(i), m)
            for sub_name, sub_m in m.named_modules():
                yield (f"{i}.{sub_name}", sub_m)
