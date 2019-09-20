import torch
import numpy as np

class SparseBinaryLayer():
    def __init__(self,inputs,outputs,density=0.1):

        #Initialize a sparse weight layer completely randomly
        self.weights = np.array(np.zeros((outputs, inputs)),dtype=bool)
        for x in range(inputs):
            for y in range(outputs):
                r = np.random.random()
                if r < density:
                    self.weights[y, x] = 1

    def forward(self,input):
        return np.bitwise_and(input,self.weights)


class SparseBinaryNeuralNetwork():
    def __init__(self,I,H,O):
        self.input_size = I
        self.l1 = SparseBinaryLayer(I,H)
        self.l2 = SparseBinaryLayer(H,O)

    def forward(self,input):
        return self.l2.forward(self.l1.forward(input))



if __name__=='__main__':
    SL = SparseBinaryLayer(5,10)
    print(SL.weights)
    input = np.array([[True,True,True,False,False]])
    print(SL.forward(input))