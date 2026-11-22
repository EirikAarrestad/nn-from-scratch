# Hovedskript: laster data, bygger nettverket, trener det og evaluerer på testsettet.
from data import load_mnist
from nn.optimizers import SGD
from nn.activations import ReLU, Softmax
from nn.layers import Dense
from nn.model import Sequential
from nn.losses import CrossEntropy

import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = load_mnist()
print(X_train.shape)

fig, axes = plt.subplots(1, 5)
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(y_train[i])
    ax.axis("off")
plt.show()

layers = [Dense(784, 128), ReLU(), Dense(128, 10), Softmax()]
model = Sequential(layers, loss_fn=CrossEntropy(), optimizer=SGD(0.1))

model.fit(X_train, y_train, epochs=10, batch_size=64)
print(model.evaluate(X_test, y_test))