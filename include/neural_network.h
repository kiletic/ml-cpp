#pragma once

#include <memory>

#include "value.h"
#include "layer.h"
#include "value_tensor.h"

struct NeuralNet {
  std::vector<std::unique_ptr<Layer>> network;

  template<class LayerType, typename... Args>
  inline NeuralNet& add(Args&& ...args) {
    this->network.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
    return *this;
  }

  std::vector<Value> get_parameters();

  ValueTensor forward(ValueTensor const&); 
  ValueTensor operator()(ValueTensor const&); 
};
