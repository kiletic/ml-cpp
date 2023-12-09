#include <vector>

#include "value.h"

struct ValueTensor {
  std::vector<std::vector<Value>> tensor;

  ValueTensor(int in_dim, int out_dim);
};
