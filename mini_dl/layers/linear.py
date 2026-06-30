import numpy as np
from mini_dl.core.module import Module
from mini_dl.core.parameter import Parameter

# Dense Layer
class Linear(Module):
    """         
    Matrix          Dimension
    Input:      (batch, in_features)
    Weight:     (in_features, out_features)
    Bias:       (out_features,)
    doutputs:(batch, out_features)
    """


    def __init__(self, in_features, out_features, w_reg_l1=0.0, w_reg_l2=0.0, b_reg_l1=0.0, b_reg_l2=0.0):
        # Cache dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Regularization hyperparameters
        self.regularization_l1 = w_reg_l1
        self.regularization_l2 = w_reg_l2
        self.bias_regularization_l1 = b_reg_l1
        self.bias_regularization_l2 = b_reg_l2

        # Initializing weights and bias
        self.weights = Parameter(
            np.random.randn(self.in_features, self.out_features)
            * np.sqrt(2.0 / self.in_features)
        )
        self.biases = Parameter(np.zeros(self.out_features))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs @ self.weights.data + self.biases.data
    
    # Backward pass
    def backward(self, doutputs):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, doutputs)
        self.dbiases = np.sum(doutputs, axis=0)

        # Gradient on values(inputs)
        self.dinputs = np.dot(doutputs, self.weights.data.T)

        # Add L1/L2 regularization gradients to the parameter gradients
        self.weights.grad = self.dweights.copy()
        self.biases.grad = self.dbiases.copy()

        if self.regularization_l1 > 0:
            self.weights.grad += self.regularization_l1 * np.sign(self.weights.data)
        if self.regularization_l2 > 0:
            self.weights.grad += self.regularization_l2 * self.weights.data

        if self.bias_regularization_l1 > 0:
            self.biases.grad += self.bias_regularization_l1 * np.sign(self.biases.data)
        if self.bias_regularization_l2 > 0:
            self.biases.grad += self.bias_regularization_l2 * self.biases.data
    
    def parameters(self):
        self.trainable_parameters = [self.weights, self.biases]    

        