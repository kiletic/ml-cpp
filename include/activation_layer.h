#pragma once

#include "layer.h"
#include "value_tensor.h"

enum ActivationFunc {
  relu, tanh, leaky_relu
};

template <ActivationFunc activation_func>
struct ActivationLayer : Layer {
  ActivationLayer() { 
    contains_params = false; 
  }

  inline std::vector<Value> get_parameters() {
    return std::vector<Value>{};
  }

  inline ValueTensor forward(ValueTensor &value_tensor) {
    for (Value &value : value_tensor) {
      if constexpr (activation_func == ActivationFunc::relu)
        value = value.relu();
      else if constexpr (activation_func == ActivationFunc::tanh)
        value = value.tanh();
      else
        value = value.leaky_relu();
    }
    return value_tensor;
  }

  inline ValueTensor operator()(ValueTensor &value_tensor) {
    return this->forward(value_tensor);
  }
};
