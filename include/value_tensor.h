#pragma once

#include <span>
#include <vector>

#include "value.h"

struct ValueTensor {
  int in_dim;
  int out_dim;
  std::vector<std::vector<Value>> tensor;

  ValueTensor(int _in_dim, int _out_dim);
  // constructor for 1-D tensor
  ValueTensor(std::initializer_list<scalar_t>);

  std::vector<Value>::iterator begin();
  std::vector<Value>::iterator end();
};
