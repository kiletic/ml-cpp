#include <cmath>
#include <random>
#include <set>
#include <memory>

#include "value.h"

Value::Value() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> d(-1, 1);

  this->internal = std::make_shared<ValueInternal>(d(gen));
} 

Value::Value(scalar_t val) {
  this->internal = std::make_shared<ValueInternal>(val);
} 

scalar_t Value::get_data() const {
  return this->internal->data;
}

scalar_t Value::get_grad() const {
  return this->internal->grad;
}

void Value::set_data(scalar_t new_data) const {
  this->internal->data = new_data;
}

// val + [number]
Value Value::operator+(scalar_t scalar) const {
  return *this + Value{scalar};
}

// val += [number]
Value& Value::operator+=(scalar_t scalar) {
  return *this = *this + Value{scalar};
}

// [number] + val
Value operator+(scalar_t scalar, Value const &val) {
  return Value{scalar} + val;
}

// val1 + val1
Value Value::operator+(Value const &other) const {
  Value ret{this->get_data() + other.get_data()};
  ret.internal->propagate_grad = [this_internal  = this->internal.get(), 
                                  other_internal = other.internal.get(),
                                  ret_internal   = ret.internal.get()]() -> void {
    other_internal->grad += ret_internal->grad; 
    this_internal->grad += ret_internal->grad; 
  };
  ret.internal->children.push_back(this->internal);
  ret.internal->children.push_back(other.internal);
  return ret;
}

// val1 += val2
Value& Value::operator+=(Value const &other) {
  return *this = *this + other;
}

// val - [number]
Value Value::operator-(scalar_t scalar) const {
  return *this + Value{-scalar};
}

// val -= [number]
Value& Value::operator-=(scalar_t scalar) {
  return *this = *this - Value{scalar};
}

// [number] - val
Value operator-(scalar_t scalar, Value const &val) {
  return Value{scalar} - val;
}

// val1 - val2
Value Value::operator-(Value const &other) const {
  Value ret{this->get_data() - other.get_data()};
  ret.internal->propagate_grad = [this_internal  = this->internal.get(), 
                                  other_internal = other.internal.get(),
                                  ret_internal   = ret.internal.get()]() -> void {
    other_internal->grad -= ret_internal->grad; 
    this_internal->grad += ret_internal->grad; 
  };
  ret.internal->children.push_back(this->internal);
  ret.internal->children.push_back(other.internal);
  return ret;
}

// val1 -= val2
Value& Value::operator-=(Value const &other) {
  return *this = *this - other;
}

// val * [number]
Value Value::operator*(scalar_t scalar) const {
  return *this * Value{scalar};
}

// val *= [number]
Value& Value::operator*=(scalar_t scalar) {
  return *this = *this * Value{scalar};
}

// [number] * val
Value operator*(scalar_t scalar, Value const &val) {
  return Value{scalar} * val;
}

// val1 * val2
Value Value::operator*(Value const &other) const {
  Value ret{this->get_data() * other.get_data()};
  ret.internal->propagate_grad = [this_internal  = this->internal.get(), 
                                  other_internal = other.internal.get(),
                                  ret_internal   = ret.internal.get()]() -> void {
    this_internal->grad += ret_internal->grad * other_internal->data; 
    other_internal->grad += ret_internal->grad * this_internal->data; 
  };
  ret.internal->children.push_back(this->internal);
  ret.internal->children.push_back(other.internal);
  return ret;
}

// val1 *= val2
Value& Value::operator*=(Value const &other) {
  return *this = *this * other;
}

// val / [number] 
Value Value::operator/(scalar_t scalar) const {
  return *this / Value{scalar};
}

// val /= [number]
Value& Value::operator/=(scalar_t scalar) {
  return *this = *this / Value{scalar};
}

// [number] / val
Value operator/(scalar_t scalar, Value const &other) {
  return Value{scalar} / other; 
}

// val1 / val2
Value Value::operator/(Value const &other) const {
  // z = x / y
  // df/dz
  // df/dx = df/dz * dz/dx
  // df/dy = df/dz * dz/dy
  // dz/dx = d(x * (y^-1)) / dx
  //       = y^-1
  // dz/dy = d(x * (y^-1)) / dy 
  //       = x * (y^-2) * -1
  //       = -x/y^2
  // x = *this
  // y = other
  Value ret{this->get_data() / other.get_data()};
  ret.internal->propagate_grad = [this_internal  = this->internal.get(), 
                                  other_internal = other.internal.get(),
                                  ret_internal   = ret.internal.get()]() -> void {
    this_internal->grad += ret_internal->grad / other_internal->data; 
    other_internal->grad += ret_internal->grad * (-this_internal->data / (other_internal->data * other_internal->data)); 
  };
  ret.internal->children.push_back(this->internal);
  ret.internal->children.push_back(other.internal);
  return ret;
}

// val1 /= val2
Value& Value::operator/=(Value const &other) {
  return *this = *this / other;
}

Value Value::sin() const {
  Value ret{std::sin(this->get_data())};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get()]() -> void {
    this_internal->grad += ret_internal->grad * std::cos(this_internal->data); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::cos() const {
  Value ret{std::cos(this->get_data())};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get()]() -> void {
    this_internal->grad += ret_internal->grad * -std::sin(this_internal->data); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::exp() const {
  auto exp_ = std::exp(this->get_data());
  Value ret{exp_};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get(),
                                  exp_]() -> void {
    this_internal->grad += ret_internal->grad * exp_; 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::tanh() const {
  auto tanh_ = std::tanh(this->get_data());
  Value ret{tanh_};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get(),
                                  tanh_sqr      = tanh_ * tanh_]() -> void {
    this_internal->grad += ret_internal->grad * (1 - tanh_sqr); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::relu() const {
  auto relu_ = std::max<scalar_t>(0, this->get_data()); 
  Value ret{relu_};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get(),
                                  relu_]() -> void {
    this_internal->grad += ret_internal->grad * (relu_ > 0 ? 1 : 0); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::leaky_relu() const {
  scalar_t alpha = 0.2;
  auto relu_ = std::max<scalar_t>(alpha * this->get_data(), this->get_data()); 
  Value ret{relu_};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get(),
                                  relu_, alpha]() -> void {
    this_internal->grad += ret_internal->grad * (relu_ > 0 ? 1 : alpha); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::log() const {
  Value ret{std::log(this->get_data())};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get()]() -> void {
      this_internal->grad += ret_internal->grad / this_internal->data; 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

Value Value::sigmoid() const {
  scalar_t sigmoid_ = 1 / (1 + std::exp(-this->get_data()));
  Value ret{sigmoid_};
  ret.internal->propagate_grad = [this_internal = this->internal.get(), 
                                  ret_internal  = ret.internal.get(),
                                  sigmoid_]() -> void {
      this_internal->grad += ret_internal->grad * sigmoid_ * (1 - sigmoid_); 
  };
  ret.internal->children.push_back(this->internal);
  return ret;
}

void Value::backward() {
  auto topological_sort = [](ValueInternal *start_node) -> std::vector<ValueInternal*> { 
    std::vector<ValueInternal*> topo;
    std::set<ValueInternal*> visited;
    auto dfs = [&topo, &visited](auto self, ValueInternal *u) -> void {
      visited.insert(u);
      for (std::shared_ptr<ValueInternal> &v : u->children) 
        if (!visited.contains(v.get()))
          self(self, v.get());
      topo.push_back(u);
    };

    dfs(dfs, start_node);
    return topo;
  };

  auto topo = topological_sort(this->internal.get());
  // maybe decouple zero_grad and backward in the future
  for (ValueInternal *u : topo)
    u->grad = 0;

  this->internal->grad = 1.0;
  std::reverse(std::begin(topo), std::end(topo));
  for (ValueInternal *u : topo)
    u->propagate_grad();
}

std::ostream& operator<<(std::ostream &out, Value const &val) {
  out << "object: " << &val << " with data: " << val.get_data() << " and grad: " << val.get_grad();
  return out;
}
