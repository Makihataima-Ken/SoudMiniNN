from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def forward(self, preds, targets):
        pass

    @abstractmethod
    def backward(self):
        pass