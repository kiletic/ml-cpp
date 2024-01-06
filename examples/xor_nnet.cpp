#include <iomanip>
#include <iostream>

#include "neural_network.h"
#include "linear_layer.h"
#include "activation_layer.h"
#include "value.h"

int main() {
  srand(time(nullptr));
  std::cout << std::setprecision(8) << std::fixed;

  NeuralNet model;
  model
    .add<LinearLayer>(2, 4)
    .add<ActivationLayer<ActivationFunc::tanh>>()
    .add<LinearLayer>(4, 4)
    .add<ActivationLayer<ActivationFunc::relu>>()
    .add<LinearLayer>(4, 2)
    .add<ActivationLayer<ActivationFunc::tanh>>()
    .add<LinearLayer>(2, 1);

  // model.initialize_weights(2);

  // model
  //   .add<LinearLayer>(2, 10)
  //   .add<ActivationLayer<ActivationFunc::tanh>>()
  //   .add<LinearLayer>(10, 5)
  //   .add<ActivationLayer<ActivationFunc::relu>>()
  //   .add<LinearLayer>(5, 1);

  auto const &params = model.get_parameters();

  scalar_t const eps = 1e-4;
  for (int its = 0; its < 100000; its++) {
    Value loss{0};
    for (int x : {0, 1}) {
      for (int y : {0, 1}) {
        ValueTensor output_tensor = model({x, y});
        Value output = output_tensor.value(); 
        Value error = ((x & y) - output);
        loss += error * error;
      }
    }

    loss = loss / 4;
    if (its % 1000 == 0)
      std::cout << "After " << its << " epochs, loss is " << loss.get_data() << std::endl;

    // zero_grad included
    loss.backward();

    for (auto const &param : params) 
      param.set_data(param.get_data() - eps * param.get_grad());
  }

  for (int x : {0, 1}) {
    for (int y : {0, 1}) {
      ValueTensor output_tensor = model({x, y});
      Value output = output_tensor.value(); 

      std::cout << "For input: " << x << " " << y << ", output is " << output.get_data() << std::endl;
    }
  }
}
