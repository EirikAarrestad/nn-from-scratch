# Sequential-modell. Setter sammen lag, og har fit() for trening, predict() for prediksjon og evaluate() for treffsikkerhet.
from nn.optimizers import SGD
from nn.losses import CrossEntropy
import numpy as np


class Sequential:
    def __init__(self, layers: list, loss_fn: CrossEntropy, optimizer: SGD) -> None:
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self) -> None:
        gradient = self.loss_fn.backward()
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 64) -> None:
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[indices], y[indices]
            losses = []
            for start in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X_shuffled[start:start + batch_size], y_shuffled[start:start + batch_size]
                y_pred = self.forward(X_batch)
                losses.append(self.loss_fn.forward(y_pred, y_batch))
                self.backward()
                self.optimizer.step(self.layers)

            print(f"Nr: {epoch + 1}: {np.mean(losses):.4f}")
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
