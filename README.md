# CUDA-Accelerated-Neural-Network

I used CUDA to accelerate a neural network written from scratch for classifying the MNIST hand-written digit dataset. The network had an input layer of 784 nerurons, a hidden layer of 64 (with tanh), and an output layer of 10 (with sigmoid). The data files are not present in the repo due to their large file sizes.

| | CPU  | GPU (CUDA) |
| ------------- | ------------- |  ------------- |
| **Training Speed**  | 57.506 sec / epoch | 3.128 sec / epoch  |
| **Test Accuracy (5 epochs)**  | 97.137%  | 96.581%  |

Clearly, the training speed was approximately 20 times faster in the CUDA implementation with comparable test accuracy.