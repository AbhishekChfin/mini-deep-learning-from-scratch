import numpy as np

class MSELoss:
        
    def forward(self, y_pred, y_true, regularization_layers=None, l1=0.0, l2=0.0):
        self.y_pred = y_pred
        self.y_true = y_true
        self.regularization_layers = regularization_layers or []
        self.l1 = l1
        self.l2 = l2
        self.loss = np.sum((self.y_pred - self.y_true)**2)/np.size(self.y_true)

        self.regularization_loss = 0.0
        for layer in self.regularization_layers:
            if hasattr(layer, "weights") and hasattr(layer.weights, "data"):
                self.regularization_loss += (
                    self.l1 * np.sum(np.abs(layer.weights.data))
                    + 0.5 * self.l2 * np.sum(layer.weights.data ** 2)
                )
        self.loss += self.regularization_loss

    def backward(self):
        self.dinputs = 2 * (self.y_pred - self.y_true) / np.size(self.y_true)

        for layer in self.regularization_layers:
            layer.regularization_l1 = self.l1
            layer.regularization_l2 = self.l2

