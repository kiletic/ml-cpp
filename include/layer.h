#pragma once

#include "value_tensor.h"

struct Layer {
  bool contains_params = true;

  virtual ~Layer() = default;

  virtual std::vector<Value> get_parameters() = 0;
  virtual ValueTensor operator()(ValueTensor&) = 0;
  virtual ValueTensor forward(ValueTensor&) = 0;
};
