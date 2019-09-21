from DataLoader import load_MNIST
from SBNN import SparseBinaryNeuralNetwork
import numpy as np
import time

if __name__ == '__main__':
    input_size = 784
    hidden_size = 500
    output_size = 10

    SBNN = SparseBinaryNeuralNetwork(input_size, hidden_size, output_size)

    X_train, Y_train, X_test, Y_test = load_MNIST()

    batch_size = 100  # max 10000
    num_classes = 10

    t1 = time.time()
    output = SBNN.forward(np.array(X_train[:batch_size], dtype=bool))
    t2 = time.time()
    correct_guesses = np.sum(np.bitwise_and(output, Y_train[:batch_size]))

    print('Example output', output[0], 'target', Y_test[0])
    print('Training accuracy', 100 * correct_guesses / (batch_size * num_classes), '%')
    print('Time elapsed', t2 - t1, 'second')
    print('Time per element', (t2 - t1) / batch_size)
