from DataLoader import load_MNIST
from SBNN import SparseBinaryNeuralNetwork_TEST
import numpy as np
import time

if __name__ == '__main__':

    input_size = 784
    hidden_size = 500
    output_size = 10

    SBNN = SparseBinaryNeuralNetwork_TEST(input_size, hidden_size, output_size)

    X_train, Y_train, X_test, Y_test = load_MNIST()

    batch_size = 5
    num_classes = 10

    t1 = time.time()
    # output = SBNN.forward_naive(X_train[:batch_size]) # 5 minutes with DTYPE = uint16, 3.4 minutes with DTYPE = bool ,optimized 0.37 minutes
    output = SBNN.forward_pytorch(X_train[:batch_size])  #
    t2 = time.time()
    output = output.cpu().numpy()
    outputargmax = np.argmax(output, axis=1)
    target = np.argmax(Y_train[:batch_size], axis=1)
    print(output, Y_train[:batch_size])
    print(outputargmax, target)
    correct_guesses = np.sum((outputargmax == target))

    print('Example output', output[0], 'target', Y_test[0])
    print('Training accuracy', 100 * correct_guesses / (batch_size), '%')
    print('Time elapsed', t2 - t1, 'second')
    print('Time per element', (t2 - t1) / batch_size)
    print('Time required for the full training+testing set', 70000 * (t2 - t1) / (batch_size), 'seconds')
