# Gradient-sjekk: sammenligner analytiske gradienter fra backward() med numeriske (finite differences).
import numpy as np
from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CrossEntropy

np.random.seed(42)

loss_fn = CrossEntropy()

X = np.random.randn(2,4)
y = np.array((0,2))

layers = [Dense(4, 5), ReLU(), Dense(5, 3), Softmax()]

def compute_loss(X: np.ndarray, y: np.ndarray) -> float:
    out = X
    for layer in layers:
        out = layer.forward(out)

    return loss_fn.forward(out, y)

compute_loss(X, y)
grad = loss_fn.backward()

for layer in reversed(layers):
    grad = layer.backward(grad)

def numerical_gradient(param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    m = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps
        loss_plus = compute_loss(X, y) 
        param[idx] = original - eps
        loss_minus = compute_loss(X, y)
        param[idx] = original
        m[idx] = (loss_plus - loss_minus) / (2 * eps)
    return m

def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b)/(np.abs(a) + np.abs(b) + 1e-12))

for idx, layer in enumerate(layers):
    if isinstance(layer, Dense):
        num_gradient_W = numerical_gradient(layer.W)
        largest_W = relative_error(num_gradient_W, layer.dW)
        
        num_gradient_b = numerical_gradient(layer.b)
        largest_b = relative_error(num_gradient_b, layer.db)
        print(f"Lag {idx}: b: {largest_b:.2e} W: {largest_W:.2e}")
