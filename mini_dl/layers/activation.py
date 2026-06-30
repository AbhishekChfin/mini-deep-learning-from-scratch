import numpy as np
from mini_dl.core.module import Module


class ReLU(Module):
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    # Backward pass
    def backward(self, doutputs):
        mask = self.inputs > 0
        self.dinputs = doutputs * mask

class Sigmoid(Module):

    @staticmethod
    def _sigmoid(inputs):
        return 1 / (1 + np.exp(-inputs))
    
    def forward(self, inputs):
        self.outputs = self._sigmoid(inputs)
    
    def backward(self, doutputs):
        self.dinputs = doutputs * (self.outputs * (1 - self.outputs))
    

class Softmax(Module):
    def forward(self, inputs):
        # shift the inputs to avoid numeric overflow
        max_values = np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs - max_values)


        self.probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = self.probabilities

    def backward(self, doutputs):
        raise NotImplementedError

        
    

        
    
