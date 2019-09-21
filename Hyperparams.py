import numpy as np

# Assume input batch is binary 0/1 in dtype DTYPE
# Because it is on 16 bits, we can not have more than 65535 links for a neuron !!!!
DTYPE = np.uint16