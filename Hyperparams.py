import numpy as np
import torch

# Assume input batch is binary 0/1 in dtype DTYPE
# Because it is on 16 bits, we can not have more than 65535 links for a neuron !!!!
DTYPE = np.int16
# For pytorch only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool.

# CUDA device
DEVICE = torch.cuda.current_device()