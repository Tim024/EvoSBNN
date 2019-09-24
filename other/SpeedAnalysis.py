import numpy as np
import time
from SBNN import *
from DataLoader import *


def test_networks():
    input_size = 784
    hidden_size = 2048
    output_size = 10

    SBNN = SparseBinaryNeuralNetwork_TEST(input_size, hidden_size, output_size)

    X_train, Y_train, X_test, Y_test = load_MNIST()

    batch_size = 1234
    num_classes = 10

    for function in [('forward_naive', SBNN.forward_naive),
                     ('forward_optimized', SBNN.forward_optimized),
                     ('forward_pytorch', SBNN.forward_pytorch),
                     ('forward_pytorch_sparse', SBNN.forward_pytorch_sparse)]:
        t1 = time.time()
        output = function[1](X_train[:batch_size])
        t2 = time.time()
        # correct_guesses = np.sum(np.bitwise_and(output.cpu().numpy(), Y_train[:batch_size]))

        print(function[0])
        # print('\tExample output', output[0], 'target', Y_test[0])
        # print('\tTraining accuracy', 100 * correct_guesses / (batch_size * num_classes), '%')
        print('\tTime elapsed', t2 - t1, 'second')
        print('\tTime per element', (t2 - t1) / batch_size)
        print('\tTime required for the full training+testing set', 70000 * (t2 - t1) / (batch_size), 'seconds')


# Numpy matrix mult analysis
def test_operations():
    SIZE = 1000
    RETRIES = 10
    TYPES = [np.int8, np.float64, bool]

    for type in TYPES:
        print('Testing type', type)
        i1 = [np.random.randn(SIZE, SIZE).astype(type) for _ in range(RETRIES)]
        i2 = [np.random.randn(SIZE, SIZE).astype(type) for _ in range(RETRIES)]
        t1 = time.time()
        for r in range(RETRIES):
            o = np.matmul(i1[r], i2[r])
        print('Time per matmul', (time.time() - t1) / RETRIES, 'seconds')

    OPERATIONS = [np.bitwise_and, np.bitwise_or, np.bitwise_xor]

    for op in OPERATIONS:
        print('Testing operation', op)
        i1 = [np.random.randn(SIZE, SIZE).astype(bool) for _ in range(RETRIES)]
        i2 = [np.random.randn(SIZE, SIZE).astype(bool) for _ in range(RETRIES)]
        t1 = time.time()
        for r in range(RETRIES):
            o = op(i1[r], i2[r])
        print('Time per operation', (time.time() - t1) / RETRIES, 'seconds')


if __name__ == '__main__':
    test_networks()
