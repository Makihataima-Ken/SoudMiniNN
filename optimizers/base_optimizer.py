from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def update(self, params, grads):
        pass