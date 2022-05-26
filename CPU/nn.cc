#include <iostream>
#include <random>
#include "nn.h"

int main()
{
    auto nn = NeuralNetwork(1);
    auto start = std::chrono::high_resolution_clock::now();
    nn.learn(5);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CPU-5 epochs: Time took: " << duration.count() << " milliseconds." << std::endl;
    return 0;
}
