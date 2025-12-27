from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, params:dict, grads:dict):
        pass