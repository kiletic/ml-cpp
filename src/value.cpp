#include "value.h"
#include <set>
#include <memory>

Value Value::operator+(Value &other) {
  Value ret{this->data + other.data};
  ret.propagate_grad = [&]() -> void {
    this->grad += ret.grad; 
    other.grad += ret.grad; 
  };
  ret.children = {this, &other};
  return ret; 
}

Value Value::operator*(Value &other) {
  Value ret{this->data * other.data};
  ret.propagate_grad = [&]() -> void {
    this->grad += other.data * ret.grad;
    other.grad += this->data * ret.grad;
  };
  ret.children = {this, &other};
  return ret; 
}

void Value::backward() {
  auto topological_sort = [&](Value *starting_value) {
    std::vector<Value*> topo;
    std::set<Value*> visited;
    auto dfs = [&](auto self, Value *u) -> void {
      for (auto v : u->children) 
        if (visited.find(v) == end(visited))
          self(self, v);
      topo.push_back(u);
    };

    dfs(dfs, starting_value);
    return topo;
  };

  this->grad = 1.0;
  auto topo = topological_sort(this);
  std::reverse(begin(topo), end(topo));
  for (auto const &u : topo)
    u->propagate_grad();
}

std::ostream& operator<<(std::ostream &out, Value const &val) {
  out << "object: " << &val << " with data: " << val.data << " and grad: " << val.grad;
  return out;
}
