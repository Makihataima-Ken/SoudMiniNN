import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    
    @abstractmethod
    def forward(self, x, training=True):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def params(self):
        return {}

    def grads(self):
        return {}