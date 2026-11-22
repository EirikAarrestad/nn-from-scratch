# Optimeringsalgoritmer (SGD, senere Adam). Oppdaterer vekter og bias i hvert lag ut fra gradientene.
from nn.layers import Dense

class SGD:
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.learning_rate = learning_rate

    def step(self, layers: list) -> None:
        for layer in layers:
            if isinstance(layer, Dense):
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db