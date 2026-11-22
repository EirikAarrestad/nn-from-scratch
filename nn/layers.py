# Lag i nettverket (Dense). Hvert lag har forward() og backward(), og lagrer vekter, bias og gradientene deres.
import numpy as np

class Dense:
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.forward_input = None
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.b = np.zeros((1, n_outputs))
        self.dW = None 
        self.db = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.forward_input = inputs
        return inputs @ self.W + self.b

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.dW = self.forward_input.T @ gradient
        self.db = np.sum(gradient, axis=0, keepdims=True)
        return gradient @ self.W.T