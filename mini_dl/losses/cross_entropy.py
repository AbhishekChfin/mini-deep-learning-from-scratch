import numpy as np
from mini_dl.layers.activation import Softmax

class CrossEntropyLoss:

    def forward(self, y_pred, y_true, regularization_layers=None, l1=0.0, l2=0.0):
        # y_pred shape: (batch_size, num_classes)
        # y_true shape: (batch_size,) or (batch_size, num_classes)
        self.regularization_layers = regularization_layers or []
        self.l1 = l1
        self.l2 = l2
        batch_size, num_classes = y_pred.shape

        # Clip so log(P) and -Y/P do not hit log(0) or divide by zero.
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Build Y, the one-hot target matrix.
        if y_true.ndim == 1:
            self.y_true = np.zeros((batch_size, num_classes))
            self.y_true[np.arange(batch_size), y_true] = 1
        elif y_true.ndim == 2:
            self.y_true = y_true
        else:
            raise ValueError("y_true must be 1D class indices or 2D one-hot labels")

        # L = -(1/B) * sum_i sum_j Y_ij * log(P_ij)
        self.loss = -np.sum(self.y_true * np.log(self.y_pred)) / batch_size

        # Add L1/L2 regularization penalty
        self.regularization_loss = 0.0
        for layer in self.regularization_layers:
            if hasattr(layer, "weights") and hasattr(layer.weights, "data"):
                self.regularization_loss += (
                    self.l1 * np.sum(np.abs(layer.weights.data))
                    + 0.5 * self.l2 * np.sum(layer.weights.data ** 2)
                )
        self.loss += self.regularization_loss

    def backward(self):
        # dL/dP = -Y / (P * B)
        batch_size = self.y_pred.shape[0]
        self.dinputs = -self.y_true / (self.y_pred * batch_size)

        for layer in self.regularization_layers:
            layer.regularization_l1 = self.l1
            layer.regularization_l2 = self.l2


class SoftmaxCrossEntropyLoss:

    def __init__(self):
        self.softmax_activation = Softmax()
        self.cross_entropy_loss = CrossEntropyLoss()

    def forward(self, logits, targets, regularization_layers=None, l1=0.0, l2=0.0):
        # logits shape: (batch_size, num_classes)
        # targets shape: (batch_size,) or (batch_size, num_classes)
        self.targets = targets
        self.regularization_layers = regularization_layers or []
        self.l1 = l1
        self.l2 = l2
        self.softmax_activation.forward(logits)
        self.probabilities = self.softmax_activation.outputs
        self.outputs = self.probabilities
        self.cross_entropy_loss.forward(
            self.probabilities,
            self.targets,
            regularization_layers=self.regularization_layers,
            l1=self.l1,
            l2=self.l2,
        )
        self.loss = self.cross_entropy_loss.loss


    def backward(self):
        # dL/dZ_norm = (softmax(z) - yi) / Batch_size
        batch_size = self.probabilities.shape[0]
        if self.targets.ndim == 2:
            target_indices = np.argmax(self.targets, axis=1)
        else:
            target_indices = self.targets

        dinputs = self.probabilities.copy()
        dinputs[np.arange(batch_size), target_indices] -= 1
        self.dinputs = dinputs / batch_size

        for layer in self.regularization_layers:
            layer.regularization_l1 = self.l1
            layer.regularization_l2 = self.l2
