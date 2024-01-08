#include <iomanip>
#include <iostream>

#include "neural_network.h"
#include "linear_layer.h"
#include "activation_layer.h"
#include "value.h"
#include "loss.h"

int main() {
  std::cout << std::setprecision(8) << std::fixed;

  NeuralNet model;
  model
    .add<LinearLayer>(2, 2)
    .add<ActivationLayer<ActivationFunc::Tanh>>()
    .add<LinearLayer>(2, 1);

  auto const params = model.get_parameters();

  scalar_t const eps = 1e-3;
  for (int its = 0; its < 100000; its++) {
    Value loss{0};
    for (int x : {0, 1}) {
      for (int y : {0, 1}) {
        ValueTensor output_tensor = model({x, y});
        loss += Loss::squared_error(output_tensor.value(), x ^ y); 
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
