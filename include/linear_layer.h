#pragma once

#include "layer.h"
#include "value_tensor.h"

struct LinearLayer : Layer {
  int in_dim;
  int out_dim;
  bool has_bias;
  ValueTensor data; 
  ValueTensor bias; 
  
  LinearLayer(int _in_dim, int _out_dim, bool has_bias = true);

  std::vector<Value> get_parameters() override; 
  
  ValueTensor forward(ValueTensor&) override;
  ValueTensor operator()(ValueTensor&) override;
};
