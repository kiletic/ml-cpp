#include <memory>

#include "neural_network.h"
#include "value.h"
#include "layer.h"

std::vector<Value> NeuralNet::get_parameters() {
  std::vector<Value> params;
  for (auto const &layer_ptr : this->network) {
    if (layer_ptr->contains_params) {
      auto const &layer_params = layer_ptr->get_parameters();
      params.insert(std::end(params), std::begin(layer_params), std::end(layer_params));
    }
  }
  return params;
}

ValueTensor NeuralNet::forward(ValueTensor const &input) {
  ValueTensor output = input;
  for (auto &layer_ptr : this->network)
    output = layer_ptr->forward(output);
  return output;
}

ValueTensor NeuralNet::operator()(ValueTensor const &input) {
  return this->forward(input);
}
