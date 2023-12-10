#include <cassert>

#include "value.h"
#include "value_tensor.h"

ValueTensor::ValueTensor(int _in_dim, int _out_dim, bool set_zero) : in_dim(_in_dim), out_dim(_out_dim) {
  // TODO: ugly set_zero hack
  this->tensor.assign(in_dim, std::vector(out_dim, (set_zero ? Value{0} : Value{})));
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

