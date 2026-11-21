# Laster og forbehandler MNIST. Lagrer en kopi i data/mnist.npz så nedlasting bare skjer én gang.

import os
import numpy as np
from sklearn.datasets import fetch_openml

CACHE = "data/mnist.npz"

def load_mnist():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        X, y = d["X"], d["y"]
    else:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")        
        y = y.astype(int)
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(CACHE, X=X, y=y)

    X = X / 255.0
    return X[:60000], X[60000:], y[:60000], y[60000:]