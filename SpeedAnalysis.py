import numpy as np
import time

# Numpy matrix mult analysis
if __name__ == '__main__':
    SIZE = 1000
    RETRIES = 10
    TYPES = [np.int8,np.float64,bool]

    for type in TYPES:
        print('Testing type',type)
        i1 = [np.random.randn(SIZE, SIZE).astype(type) for _ in range(RETRIES)]
        i2 = [np.random.randn(SIZE, SIZE).astype(type) for _ in range(RETRIES)]
        t1 = time.time()
        for r in range(RETRIES):
            o = np.matmul(i1[r],i2[r])
        print('Time per matmul',(time.time()-t1)/RETRIES,'second')

    OPERATIONS = [np.bitwise_and,np.bitwise_or,np.bitwise_xor]

    for op in OPERATIONS:
        print('Testing operation', op)
        i1 = [np.random.randn(SIZE, SIZE).astype(bool) for _ in range(RETRIES)]
        i2 = [np.random.randn(SIZE, SIZE).astype(bool) for _ in range(RETRIES)]
        t1 = time.time()
        for r in range(RETRIES):
            o = op(i1[r],i2[r])
        print('Time per matmul', (time.time() - t1) / RETRIES, 'second')

    # To get GPU support for these simple numpy operations, pytorch is maybe not optimal. Maybe minpy ?


