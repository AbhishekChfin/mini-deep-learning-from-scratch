from mini_dl.core.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            layer.forward(outputs)
            outputs = layer.outputs
        self.outputs = outputs
    
    def backward(self, doutputs):
        dinputs = doutputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs
        self.dinputs = dinputs
    
    def parameters(self):
        self.trainable_parameters = []
        for layer in self.layers:
            layer.parameters()
            self.trainable_parameters.extend(layer.trainable_parameters)
    
