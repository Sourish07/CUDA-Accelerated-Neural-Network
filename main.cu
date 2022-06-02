#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include "nn.cuh"
#include "timer.h"

int main()
{
    auto nn = MNISTNeuralNetwork();
    start_timer();
    nn.learn(1);
    end_timer();
    return 0;
    
}
