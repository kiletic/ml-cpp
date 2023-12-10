#pragma once

#include "value_tensor.h"

struct Layer {
  virtual ~Layer() = default;
  virtual ValueTensor operator()(ValueTensor&) = 0;
  virtual ValueTensor forward(ValueTensor&) = 0;
};
