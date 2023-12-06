#include <iostream>

#include "value.h"

int main() {
  Value v{2.0};
  Value u(3.0);
  Value res = u * (u + v);
  // Value res = u + v;
  // Value res = uv * u;

  res.backward();
  std::cout << res << std::endl;
  // std::cout << u + v << std::endl;
  std::cout << u << std::endl;
  std::cout << v << std::endl;
}
