# Aktiveringsfunksjoner (ReLU, Sigmoid, Softmax). Hver har forward() og backward(), men ingen vekter.
import numpy as np

class ReLU:
    def __init__(self) -> None:
        self.forward_input = None 

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.forward_input = inputs
        return np.maximum(0, self.forward_input)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        mask = self.forward_input > 0
        return gradient * mask

class Sigmoid:
    def __init__(self) -> None:
        self.forward_output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = 1 / (1 + np.exp(-inputs))
        self.forward_output = output
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        return gradient * self.forward_output * (1 - self.forward_output)

class Softmax:
    def __init__(self) -> None:
        self.forward_output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        output = exps / np.sum(exps, axis=1, keepdims=True)
        self.forward_output = output
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        # Gradienten regnes ut i losses.py, trenger ikke å gjøre noe her
        return gradient
