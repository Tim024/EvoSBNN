# EvoSBNN
Evolutionary Algorithm Applied to Sparse Binary Neural Network Search

+ SBNN.py contains the code of the neural network and the binary layers.
+ Random Search contains an example with a small network running MNIST.

# Speed test results:

- Batch of size 1234
- Neural network 784x2048x2048x10
- MNIST

```

Loading MNIST...
Test set shape (10000, 784) (10000, 10)
Train set shape (60000, 784) (60000, 10)

Hyperparams:
	Using dtype for numpy: <class 'numpy.int16'> for pytorch: torch.float32
	Using device cuda:0
forward_naive
	Time elapsed 34.9804892539978 second
	Time per element 0.028347236024309403
	Time required for the full training+testing set 1984.306521701658 seconds
forward_optimized
	Time elapsed 5.349349021911621 second
	Time per element 0.004334966792472951
	Time required for the full training+testing set 303.44767547310653 seconds
forward_pytorch
	Time elapsed 0.03110790252685547 second
	Time per element 2.520899718545824e-05
	Time required for the full training+testing set 1.7646298029820768 seconds
forward_pytorch_sparse
	Time elapsed 0.4881618022918701 second
	Time per element 0.00039559303265143447
	Time required for the full training+testing set 27.69151228560041 seconds

  Hyperparams:
  	Using dtype for numpy: <class 'bool'> for pytorch: torch.float32
  	Using device cpu
  forward_naive
  	Time elapsed 35.39233708381653 second
  	Time per element 0.028680986291585518
  	Time required for the full training+testing set 2007.669040410986 seconds
  forward_optimized
  	Time elapsed 5.363753080368042 second
  	Time per element 0.004346639449244767
  	Time required for the full training+testing set 304.26476144713365 seconds
  forward_pytorch
  	Time elapsed 0.12663578987121582 second
  	Time per element 0.0001026221960058475
  	Time required for the full training+testing set 7.183553720409325 seconds
  forward_pytorch_sparse
  	Time elapsed 32.43106293678284 second
  	Time per element 0.02628125035395692
  	Time required for the full training+testing set 1839.6875247769842 seconds

    Hyperparams:
    	Using type for numpy: <class 'numpy.int16'> for pytorch: torch.int16
    	Using device cpu
    forward_naive
    	Time elapsed 34.786320209503174 second
    	Time per element 0.02818988671758766
    	Time required for the full training+testing set 1973.2920702311362 seconds
    forward_optimized
    	Time elapsed 5.4067463874816895 second
    	Time per element 0.004381480054685324
    	Time required for the full training+testing set 306.70360382797264 seconds
    forward_pytorch
    	Time elapsed 4.248442888259888 second
    	Time per element 0.0034428224378119025
    	Time required for the full training+testing set 240.99757064683317 seconds
    forward_pytorch_sparse
    	Time elapsed 26.95875072479248 second
    	Time per element 0.02184663754035047
    	Time required for the full training+testing set 1529.264627824533 seconds

  ```
