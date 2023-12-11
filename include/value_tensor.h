#pragma once

#include <span>
#include <variant>
#include <vector>

#include "value.h"

struct ValueTensor {
  int in_dim;
  int out_dim;
  std::vector<std::vector<Value>> tensor;

  ValueTensor() : in_dim(0), out_dim(0), tensor({}) {}
  ValueTensor(int _in_dim, int _out_dim, bool set_zero = false);

  // constructor for 1-D tensor
  template<typename T>
  inline ValueTensor(std::initializer_list<T> init_list) {
    static_assert(std::is_integral_v<T>, "1-D tensor needs to have integral type");
    if constexpr (std::is_same_v<T, scalar_t>)
      this->tensor.assign(1, std::vector<Value>{std::begin(init_list), std::end(init_list)});
    else {
      std::vector<scalar_t> converted_list{std::begin(init_list), std::end(init_list)};
      this->tensor.assign(1, std::vector<Value>{std::begin(converted_list), std::end(converted_list)});
    }
    
    this->in_dim = 1;
    this->out_dim = init_list.size();
  }

  Value value() const;
  std::vector<Value>& operator[](size_t);
  std::vector<Value>::iterator begin();
  std::vector<Value>::iterator end();
};
