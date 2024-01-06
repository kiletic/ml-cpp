#include <iostream>

#include "neural_network.h"
#include "linear_layer.h"
#include "activation_layer.h"

int main() {
  {
    Value a = 2;
    Value b = 3;
    Value v = a + b + 1;
    std::cout << v.internal.use_count() << std::endl;
    std::cout << a.internal.use_count() << std::endl;
    std::cout << b.internal.use_count() << std::endl;
  }

  {
    Value a = 2000;
    Value b = 3000;
    Value v = a + b + 5;
  }
}
