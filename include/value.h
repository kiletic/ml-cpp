#pragma once

#include <functional>
#include <iostream>
#include <memory>

using scalar_t = float;

struct ValueInternal {
  scalar_t data;
  scalar_t grad;
  std::function<void()> propagate_grad;  
  std::vector<std::shared_ptr<ValueInternal>> children; 

  ValueInternal(scalar_t _data) : data(_data), grad(0), propagate_grad([](){}), children({}) {}
};

struct Value {
  std::shared_ptr<ValueInternal> internal;

  Value();
  Value(scalar_t);

  Value operator+(scalar_t scalar) const;
  Value operator+(Value const &other) const;
  Value& operator+=(scalar_t scalar);
  Value& operator+=(Value const &other);

  Value operator-(scalar_t scalar) const;
  Value operator-(Value const &other) const;
  Value& operator-=(scalar_t scalar);
  Value& operator-=(Value const &other);

  Value operator*(scalar_t scalar) const;
  Value operator*(Value const &other) const;
  Value& operator*=(scalar_t scalar);
  Value& operator*=(Value const &other);

  Value operator/(scalar_t scalar) const;
  Value operator/(Value const &other) const;
  Value& operator/=(scalar_t scalar);
  Value& operator/=(Value const &other);

  Value sin() const;
  Value cos() const;
  Value exp() const;
  Value log() const;
  Value tanh() const;
  Value relu() const;
  Value sigmoid() const;
  Value leaky_relu() const;

  scalar_t get_data() const;
  scalar_t get_grad() const;
  void set_data(scalar_t) const;
  void backward();

  friend std::ostream& operator<<(std::ostream& out, Value const &val); 
};

Value operator+(scalar_t scalar, Value const &val); 
Value operator-(scalar_t scalar, Value const &val); 
Value operator*(scalar_t scalar, Value const &val); 
Value operator/(scalar_t scalar, Value const &val); 
