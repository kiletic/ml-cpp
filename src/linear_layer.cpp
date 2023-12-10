#include "linear_layer.h"

LinearLayer::LinearLayer(int in_dim, int out_dim) : data(in_dim, out_dim) {}

ValueTensor LinearLayer::forward(ValueTensor &value_tensor) {
  return value_tensor;
}

ValueTensor LinearLayer::operator()(ValueTensor &value_tensor) {
  return this->forward(value_tensor);
}
