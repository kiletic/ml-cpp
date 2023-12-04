#pragma once
#include <functional>
#include <iostream>
#include <memory>

using scalar_t = double;

struct Value {
  scalar_t data;
  scalar_t grad;
  std::function<void()> propagate_grad;  
  std::vector<Value*> children; 

  Value(scalar_t _data) : data(_data), grad(0.0), propagate_grad([](){}) {}

  Value operator+(Value &other);
  Value operator*(Value &other);

  void backward();

  friend std::ostream& operator<<(std::ostream& out, Value const &val); 
};
