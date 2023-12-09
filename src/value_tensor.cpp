#include "value.h"
#include "value_tensor.h"

ValueTensor::ValueTensor(int in_dim, int out_dim) {
  this->tensor.assign(in_dim, std::vector(out_dim, Value{}));
}
