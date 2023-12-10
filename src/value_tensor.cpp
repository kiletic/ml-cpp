#include <cassert>

#include "value.h"
#include "value_tensor.h"

ValueTensor::ValueTensor(int _in_dim, int _out_dim) : in_dim(_in_dim), out_dim(_out_dim) {
  this->tensor.assign(in_dim, std::vector(out_dim, Value{}));
}

ValueTensor::ValueTensor(std::initializer_list<scalar_t> s) {
  this->tensor.assign(1, std::vector<Value>{std::begin(s), std::end(s)});
  this->in_dim = 1;
  this->out_dim = s.size();
}

std::vector<Value>::iterator ValueTensor::begin() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return std::begin(this->tensor[0]);
}

std::vector<Value>::iterator ValueTensor::end() {
  assert(this->tensor.size() == 1 && "Only 1-D tensors are iterable (for now)");
  return std::end(this->tensor[0]);
}
