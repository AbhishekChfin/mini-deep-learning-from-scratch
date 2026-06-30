import numpy as np
from mini_dl.core.module import Module


class Layer_dropout(Module):
    def __init__(self, dropout_rate=0.5, training=True):
        self.dropout_rate = dropout_rate
        self.training = training

    def forward(self, inputs):
        self.inputs = inputs

        if not self.training:
            self.outputs = inputs
            self.mask = np.ones_like(inputs)
            return

        keep_prob = 1.0 - self.dropout_rate
        self.mask = (np.random.rand(*inputs.shape) < keep_prob).astype(inputs.dtype)
        self.outputs = inputs * self.mask / keep_prob

    def backward(self, doutputs):
        if not self.training:
            self.dinputs = doutputs
            return

        self.dinputs = doutputs * self.mask / (1.0 - self.dropout_rate)
