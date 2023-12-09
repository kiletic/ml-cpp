#pragma once

#include "value_tensor.h"

struct Layer {
  virtual ~Layer();
  virtual ValueTensor operator()() = 0;
};

struct LinearLayer : Layer {
  int in_dim;
  int out_dim;
  ValueTensor data; 
  
  LinearLayer(int _in_dim, int _out_dim);
  
  ValueTensor operator()() override;
};
