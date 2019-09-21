import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import cm
from Hyperparams import *


def load_MNIST():
    print("Loading MNIST...")
    mat = sio.loadmat('./data/mnist.mat')
    num_classes = 10
    # Binarize data
    X_train = mat['trainX'] / 254
    X_train[X_train > 0.3] = 1
    X_train[X_train < 0.31] = 0
    X_test = mat['testX'] / 254
    X_test[X_test > 0.3] = 1
    X_test[X_test < 0.31] = 0
    # Y in one hot
    Y_train = np.array(
        [[1 if i == mat['trainY'][0][c] else 0 for i in range(num_classes)] for c in range(len(mat['trainY'][0]))])
    Y_test = np.array(
        [[1 if i == mat['testY'][0][c] else 0 for i in range(num_classes)] for c in range(len(mat['testY'][0]))])

    print("Test set shape", X_test.shape, Y_test.shape)
    print("Train set shape", X_train.shape, Y_train.shape)

    return np.array(X_train,dtype=DTYPE), np.array(Y_train,dtype=DTYPE), np.array(X_test,dtype=DTYPE), np.array(Y_test,dtype=DTYPE)


if __name__ == '__main__':
    X_train, Y_train, _, _ = load_MNIST()

    # Show example:
    r = np.random.randint(0, len(X_train))
    print(Y_train[r])
    plt.imshow(X_train[r].reshape((28, 28)), cmap=cm.binary)  # white is 0, black is 1 with cm.binary
    plt.show()
