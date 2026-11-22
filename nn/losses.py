# Tapsfunksjoner (Cross-entropy, MSE). Regner ut tapet og gradienten med hensyn på prediksjonene.
import numpy as np

def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    m = np.zeros((labels.shape[0], n_classes))
    m[np.arange(labels.shape[0]), labels] = 1
    return m

class CrossEntropy:
    def __init__(self) -> None:
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = one_hot(y_true)

        clipped = np.clip(self.y_pred, a_min=1e-12, a_max=1)
        per_sample_loss = np.sum(np.log(clipped) * self.y_true, axis=1)
        return -np.mean(per_sample_loss)

    def backward(self) -> np.ndarray:
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]