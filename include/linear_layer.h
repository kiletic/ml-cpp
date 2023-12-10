#pragma once

#include "layer.h"
#include "value_tensor.h"

struct LinearLayer : Layer {
  int in_dim;
  int out_dim;
  ValueTensor data; 
  
  LinearLayer(int _in_dim, int _out_dim);
  
  ValueTensor forward(ValueTensor&) override;
  ValueTensor operator()(ValueTensor&) override;
};
