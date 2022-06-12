#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include "nn.cuh"
#include "timer.h"

int main()
{
    auto nn = MNISTNeuralNetwork();
    
    auto timer = Timer("Network-GPU");
    nn.learn(5);
    timer.stop();

    return 0;
}
