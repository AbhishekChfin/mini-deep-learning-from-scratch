from abc import ABC, abstractmethod
class Module(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, doutputs):
        pass

    def parameters(self):
        self.trainable_parameters = []

    def zero_grad(self):
        for parameter in self.trainable_parameters:
            parameter.zero_grad()
