from DataLoader import load_MNIST
from SBNN import SparseBinaryNeuralNetwork
import numpy as np
import time

if __name__ == '__main__':

    input_size = 784
    hidden_size = 500
    output_size = 10

    SBNN = SparseBinaryNeuralNetwork(input_size,hidden_size,output_size)

    X_train, Y_train, X_test, Y_test = load_MNIST(only_half=True)
    length = len(X_test)

    t1 = time.time()
    output = SBNN.forward(np.array(X_test,dtype=bool))
    t2 = time.time()
    correct_guesses = np.bitwise_and(output,Y_test)

    print('Training accuracy',np.sum(correct_guesses))
    print('Time elapsed',t2-t1,'second')
    print('Time per element',(t2-t1)/length)