from data import load_mnist
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = load_mnist()
print(X_train.shape)

fig, axes = plt.subplots(1, 5)
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap="gray")
    ax.set_title(y_train[i])
    ax.axis("off")
plt.show()