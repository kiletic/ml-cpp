#include <set>
#include <memory>

#include "value.h"

Value::Value(scalar_t val) {
  this->internal = std::make_shared<ValueInternal>(val);
} 

scalar_t Value::get_data() const {
  return this->internal->data;
}

scalar_t Value::get_grad() const {
  return this->internal->grad;
}

Value Value::operator+(Value const &other) {
  Value ret{this->get_data() + other.get_data()};
  // TODO: why is it memory leaking when capturing without get()?
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

Value Value::operator*(Value const &other) {
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

void Value::backward() {
  auto topological_sort = [](ValueInternal *starting_value) -> std::vector<ValueInternal*> { 
    std::vector<ValueInternal*> topo;
    std::set<ValueInternal*> visited;
    auto dfs = [&](auto self, ValueInternal *u) -> void {
      visited.insert(u);
      for (std::shared_ptr<ValueInternal> v : u->children) 
        if (visited.find(v.get()) == end(visited))
          self(self, v.get());
      topo.push_back(u);
    };

    dfs(dfs, starting_value);
    return topo;
  };

  this->internal->grad = 1.0;
  auto topo = topological_sort(this->internal.get());
  std::reverse(begin(topo), end(topo));
  for (ValueInternal *u : topo)
    u->propagate_grad();
}

std::ostream& operator<<(std::ostream &out, Value const &val) {
  out << "object: " << &val << " with data: " << val.get_data() << " and grad: " << val.get_grad();
  return out;
}
