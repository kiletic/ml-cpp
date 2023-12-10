#include <iostream>

#include "neural_network.h"
#include "linear_layer.h"
#include "activation_layer.h"

int main() {
  NeuralNet model;
  model
    .add<LinearLayer>(2, 3)
    .add<ActivationLayer<ActivationFunc::tanh>>()
    .add<LinearLayer>(3, 2)
    .add<ActivationLayer<ActivationFunc::relu>>()
    .add<LinearLayer>(2, 1);

  auto output = model({1, 2, 3});
  for (auto const &x : output) {
    std::cout << x << std::endl;
  }
}
