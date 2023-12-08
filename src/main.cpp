#include <iostream>

#include "value.h"

int main() {
  Value v{2.0};
  Value u(3.0);
  Value temp = 1 + u;
  // Value res = 1 + u * 2 * (u + v);

  // res.backward();
  // std::cout << res << std::endl;
  std::cout << u << std::endl;
  std::cout << v << std::endl;
}
