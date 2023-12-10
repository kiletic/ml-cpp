#include <memory>

#include "neural_network.h"
#include "value.h"
#include "layer.h"

ValueTensor NeuralNet::forward(ValueTensor &input) {
  for (auto &layer_ptr : this->network)
    input = layer_ptr->forward(input);
  return input;
}

ValueTensor NeuralNet::operator()(ValueTensor &input) {
  return this->forward(input);
}
