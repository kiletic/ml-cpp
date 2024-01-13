#include <cassert>

#include "value.h"
#include "value_tensor.h"

ValueTensor::ValueTensor(int _in_dim, int _out_dim, bool set_zero) : in_dim(_in_dim), out_dim(_out_dim) {
  for (int i = 0; i < in_dim; i++)
    this->tensor.push_back(std::vector<Value>(out_dim));

  if (set_zero) {
    for (int i = 0; i < in_dim; i++)
      for (int j = 0; j < out_dim; j++)
        this->tensor[i][j].set_data(0);
  }
}

Value ValueTensor::value() const {
  assert(this->in_dim == 1 && this->out_dim == 1 && "Cannot fetch Value from ValueTensor that has > 1 of them.");
  return this->tensor[0][0];
}

std::vector<Value>& ValueTensor::operator[](size_t pos) {
  return this->tensor[pos];
}

std::vector<Value>::iterator ValueTensor::begin() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return std::begin(this->tensor[0]);
}

std::vector<Value>::iterator ValueTensor::end() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return std::end(this->tensor[0]);
}

