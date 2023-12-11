#include "linear_layer.h"
#include <cassert>

LinearLayer::LinearLayer(int in_dim, int out_dim, bool has_bias) : data(in_dim, out_dim) {
  if (has_bias)
    bias = ValueTensor(1, out_dim);
}

ValueTensor LinearLayer::forward(ValueTensor &value_tensor) {
  // need to be chained
  assert(value_tensor.out_dim == this->data.in_dim);
  
  ValueTensor output(value_tensor.in_dim, this->data.out_dim, /*set_zero=*/ true);
  for (int i = 0; i < value_tensor.in_dim; i++)
    for (int j = 0; j < this->data.out_dim; j++)
      for (int k = 0; k < this->data.in_dim; k++)
        output[i][j] += value_tensor[i][k] * this->data[k][j]; 
  
  for (int i = 0; i < output.in_dim; i++)
    for (int j = 0; j < output.out_dim; j++)
      output[i][j] += bias[0][j];

  return output;
}

std::vector<Value> LinearLayer::get_parameters() {
  std::vector<Value> params(this->data.in_dim * this->data.out_dim + bias.out_dim);
  for (int i = 0; i < this->data.in_dim; i++)
    for (int j = 0; j < this->data.out_dim; j++)
      params[j + i * this->data.out_dim] = this->data[i][j];
  
  for (int i = 0; i < this->bias.out_dim; i++)
    params[this->data.in_dim * this->data.out_dim + i] = bias[0][i];

  return params;
}

ValueTensor LinearLayer::operator()(ValueTensor &value_tensor) {
  return this->forward(value_tensor);
}
