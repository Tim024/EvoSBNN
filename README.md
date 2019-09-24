# EvoSBNN
Evolutionary Algorithm Applied to Sparse Binary Neural Network Search

+ SBNN.py contains the code of the neural network and the binary layers.
+ Random Search contains an example with a small network running MNIST.

# Speed test results:

- Batch of size 1234
- Neural network 784x1000x1000x10
- MNIST

```
- forward_naive
    Time elapsed 15.327361345291138 second
	Time per element 0.0124208762927805
	Time required for the full training+testing set 869.461340494635 seconds
- forward_optimized
	Time elapsed 1.757552146911621 second
	Time per element 0.0014242724043043932
	Time required for the full training+testing set 99.69906830130752 seconds
- forward_pytorch
	Time elapsed 0.017002582550048828 second
	Time per element 1.3778429943313474e-05
	Time required for the full training+testing set 0.9644900960319432 seconds
  ```
