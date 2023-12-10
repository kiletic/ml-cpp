#include <cassert>

#include "value.h"
#include "value_tensor.h"

ValueTensor::ValueTensor(int in_dim, int out_dim) {
  this->tensor.assign(in_dim, std::vector(out_dim, Value{}));
}

std::vector<Value>::iterator ValueTensor::begin() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return this->tensor[0].begin();
}

std::vector<Value>::iterator ValueTensor::end() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return this->tensor[0].end();
}
