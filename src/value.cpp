#include "value.h"
#include <set>
#include <memory>

Value Value::operator+(Value &other) {
  Value ret{this->data + other.data};
  ret.propagate_grad = [&]() -> void {
    this->grad += ret.grad; 
    other.grad += ret.grad; 
  };
  ret.children = {std::make_shared<Value>(*this), std::make_shared<Value>(other)};
  return ret; 
}

Value Value::operator*(Value &other) {
  Value ret{this->data * other.data};
  ret.propagate_grad = [&]() -> void {
    this->grad += other.data * ret.grad;
    other.grad += this->data * ret.grad;
  };
  ret.children = {std::make_shared<Value>(*this), std::make_shared<Value>(other)};
  return ret; 
}

void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo;
  std::set<std::shared_ptr<Value>> visited;
  auto topological_sort = [&](auto self, std::shared_ptr<Value> u) -> void {
    for (auto v : u->children) 
      if (visited.find(v) == end(visited))
        self(self, v);
    topo.push_back(u);
  };

  this->grad = 1.0;
  topological_sort(topological_sort, std::make_shared<Value>(*this));
  std::reverse(begin(topo), end(topo));
  for (auto const &u : topo)
    u->propagate_grad();
}

std::ostream& operator<<(std::ostream &out, Value const &val) {
  out << "object: " << &val << " with data: " << val.data << " and grad: " << val.grad;
  return out;
}
