import torch
import numpy as np

class SparseBinaryLayer():
    def __init__(self,inputs,outputs,density=0.1):
        super(SparseBinaryLayer, self).__init__()

        #Initialize a sparse weight layer completely randomly
        self.weights = np.array(np.zeros((inputs, outputs)),dtype = bool)
        for x in range(inputs):
            for y in range(outputs):
                r = np.random.random()
                if r < density:
                    self.weights[x, y] = 1

        print(self.weights)


if __name__=='__main__':
    SL = SparseBinaryLayer(5,10)