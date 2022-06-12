#include <iostream>
#include <random>
#include "nn.h"
#include "../timer.h"

int main()
{
    auto nn = NeuralNetwork();

    auto timer = Timer("Network-CPU");
    nn.learn(1);
    timer.stop();

    return 0;
}
