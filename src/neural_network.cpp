#include <memory>

#include "neural_network.h"
#include "value.h"
#include "layer.h"

ValueTensor NeuralNet::forward(ValueTensor const &input) {
  ValueTensor output = input;
  for (auto &layer_ptr : this->network)
    output = layer_ptr->forward(output);
  return output;
}

ValueTensor NeuralNet::operator()(ValueTensor const &input) {
  return this->forward(input);
}
