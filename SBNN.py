import torch
import numpy as np

class SparseBinaryLayer:
    def __init__(self, input_size, output_size, density=0.1, op=np.bitwise_and):

        # Initialize a sparse weight layer completely randomly
        self.operation = op
        self.weights = np.array(np.zeros((output_size, input_size)), dtype=bool) # Row = output neuron, Column = input neuron, weight(out,in) = True if link
        for x in range(input_size):
            for y in range(output_size):
                r = np.random.random()
                if r < density:
                    self.weights[y, x] = 1

    def forward(self, input_batch):
        # TODO this is clearly too slow...
        output_batch = []
        for input in input_batch:
            output = []
            for neuron in self.weights:
                output.append(True if np.sum(self.operation(input, neuron)) > 2 else False) # Activate only if at least 2 inputs are true
                # print(neuron,'x', input, '->',output[-1])
            output_batch.append(output)
        return np.array(output_batch)


class SparseBinaryNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.l1 = SparseBinaryLayer(input_size, hidden_size)
        self.l2 = SparseBinaryLayer(hidden_size, output_size)

    def forward(self, input):
        return self.l2.forward(self.l1.forward(input))


if __name__ == '__main__':
    SL = SparseBinaryLayer(5, 10)
    print(SL.weights)
    input = np.array([[True, True, True, True, True],[False, False, False, False, False]])
    print(SL.forward(input))
