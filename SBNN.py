import torch
import numpy as np
from Hyperparams import *


class SparseBinaryLayer:
    def __init__(self, input_size, output_size, density=0.1):

        # Initialize a sparse weight layer completely randomly
        self.weights = np.array(np.zeros((output_size, input_size)),
                                dtype=DTYPE)  # Row = output neuron, Column = input neuron, weight(out,in) = 1 if link

        for x in range(input_size):
            for y in range(output_size):
                r = np.random.random()
                if r < density:
                    self.weights[y, x] = 1

    def forward_naive(self, input_batch):
        # TODO this is clearly too slow...
        output_batch = []
        for input in input_batch:
            output = []
            for neuron in self.weights:
                output.append(True if np.sum(
                    np.bitwise_and(input, neuron)) > 2 else False)  # Activate only if at least 2 inputs are true
                # print(neuron,'x', input, '->',output[-1])
            output_batch.append(output)
        return np.array(output_batch)

    def forward_optimized(self, input_batch):
        output = np.matmul(input_batch,self.weights.T)
        output[output < 3] = 0 # Same activation as before
        output[output > 2] = 1
        return output



class SparseBinaryNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.l1 = SparseBinaryLayer(input_size, hidden_size)
        self.l2 = SparseBinaryLayer(hidden_size, output_size)

    def forward_naive(self, input):
        return self.l2.forward_naive(self.l1.forward_naive(input))

    def forward_optimized(self, input):
        return self.l2.forward_optimized(self.l1.forward_optimized(input))


if __name__ == '__main__':
    SL = SparseBinaryLayer(5, 10, density=0.5)
    print(SL.weights)
    input = np.array([[True, True, True, True, True], [False, False, False, False, False]])
    print(SL.forward_naive(input))
    print(SL.forward_optimized(input))
