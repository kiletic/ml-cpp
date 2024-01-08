#pragma once

#include "layer.h"
#include "value_tensor.h"

enum ActivationFunc {
  Relu, 
  Tanh, 
  Leaky_relu,
  Sigmoid
};

template <ActivationFunc activation_func>
struct ActivationLayer : Layer {
  ActivationLayer() { 
    contains_params = false; 
  }

  constexpr std::vector<Value> get_parameters() override {
    return std::vector<Value>{};
  }

  inline ValueTensor forward(ValueTensor &value_tensor) {
    for (Value &value : value_tensor) {
      if constexpr (activation_func == ActivationFunc::Relu)
        value = value.relu();
      else if constexpr (activation_func == ActivationFunc::Tanh)
        value = value.tanh();
      else if constexpr (activation_func == ActivationFunc::Sigmoid)
        value = value.sigmoid();
      else
        value = value.leaky_relu();
    }
    return value_tensor;
  }

  inline ValueTensor operator()(ValueTensor &value_tensor) {
    return this->forward(value_tensor);
  }
};
