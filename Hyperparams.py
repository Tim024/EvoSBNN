import numpy as np
import torch

print("Hyperparams:")
# Because it is on 16 bits, we can not have more than 65535 links for a neuron !!!!
DTYPE = np.int16
DTYPEP = torch.float32  # GPU support not implemented for int, Sparse only implemented for 32 Float
print("\tUsing type for numpy:", DTYPE,"for pytorch:", DTYPEP)

# CUDA device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\tUsing device", DEVICE)


def create_tensor(numpy_array):
    if DTYPEP == torch.float16:
        return torch.HalfTensor(numpy_array)
    if DTYPEP == torch.float32:
        return torch.FloatTensor(numpy_array)
    if DTYPEP == torch.int16 or DTYPEP == torch.short:
        return torch.ShortTensor(numpy_array)
    if DTYPEP == torch.bool:
        return torch.BoolTensor(numpy_array)
    print("Error, type unsupported in Hyperparams", DTYPEP)
    return None
