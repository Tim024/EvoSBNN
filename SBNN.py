import torch
import numpy as np
from Hyperparams import *


class SparseBinaryLayer_TEST:
    def __init__(self, input_size, output_size, density=0.1):

        # Initialize a sparse weight layer completely randomly
        self.weights = np.array(np.zeros((output_size, input_size)),
                                dtype=DTYPE)  # Row = output neuron, Column = input neuron, weight(out,in) = 1 if link

        for x in range(input_size):
            for y in range(output_size):
                r = np.random.random()
                if r < density:
                    self.weights[y, x] = 1

        self.weights_GPU = torch.FloatTensor(self.weights).to(DEVICE)

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

    def forward_pytorch(self,input_batch):
        output = torch.matmul(input_batch,self.weights_GPU.t())
        output[output < 3] = 0  # Same activation as before
        output[output > 2] = 1
        return output


class SparseBinaryNeuralNetwork_TEST:
    def __init__(self, input_size, hidden_size, output_size):
        self.l1 = SparseBinaryLayer_TEST(input_size, hidden_size)
        self.l2 = SparseBinaryLayer_TEST(hidden_size, hidden_size)
        self.l3 = SparseBinaryLayer_TEST(hidden_size, output_size)

    def forward_naive(self, input):
        return self.l3.forward_naive(self.l2.forward_naive(self.l1.forward_naive(input)))

    def forward_optimized(self, input):
        return self.l3.forward_optimized(self.l2.forward_optimized(self.l1.forward_optimized(input)))

    def forward_pytorch(self, input):
        # print(input)
        input = torch.FloatTensor(input).to(DEVICE)
        o = self.l1.forward_pytorch(input)
        o = self.l2.forward_pytorch(o)
        o = self.l3.forward_pytorch(o)
        return o


if __name__ == '__main__':
    SL = SparseBinaryLayer_TEST(5, 10, density=0.5)
    print(SL.weights)
    input = np.array([[True, True, True, True, True], [False, False, False, False, False]])
    print(SL.forward_naive(input))
    print(SL.forward_optimized(input))
    print(SL.forward_pytorch(torch.FloatTensor(input).to(DEVICE)))
