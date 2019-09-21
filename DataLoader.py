import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import cm

def load_MNIST(only_half=False):
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

    if (only_half):
        X_train = X_train[:int(len(X_train)/2)]
        Y_train = Y_train[:int(len(Y_train)/2)]
        X_test = X_test[:int(len(X_test)/2)]
        Y_test = Y_test[:int(len(Y_test)/2)]

    print("Test set shape", X_test.shape, Y_test.shape)
    print("Train set shape", X_train.shape, Y_train.shape)

    return X_train,Y_train,X_test,Y_test


if __name__ == '__main__':

    X_train,Y_train,_,_ = load_MNIST()

    # Show example:
    r = np.random.randint(0, len(X_train))
    print(Y_train[r])
    plt.imshow(X_train[r].reshape((28, 28)), cmap=cm.binary)  # white is 0, black is 1 with cm.binary
    plt.show()