#include <iostream>
#include "value.h"

int main() {
  Value v{2.0};
  Value u(3.0);
  Value uv = u + v; 
  Value res = uv * u;

  res.backward();
  std::cout << res << std::endl;
  std::cout << uv << std::endl;
  std::cout << u << std::endl;
  std::cout << v << std::endl;
}
