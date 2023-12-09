#include <iomanip>
#include <iostream>

#include "value.h"

int main() {
  srand(time(nullptr));
  std::cout << std::setprecision(8) << std::fixed;

  Value w11{}, w12{}, w13{};
  Value w21{}, w22{};
  Value w31{}, w32{};
  Value b1{}, b2{}, b3{};

  auto forward = [&](Value const &x, Value const &y) -> Value {
    Value W11 = (w11 * x + w11 * y + b1).tanh();
    Value W12 = (w12 * x + w12 * y + b1).tanh();
    Value W13 = (w13 * x + w13 * y + b1).tanh();

    Value W21 = (w21 * W11 + w21 * W12 + w21 * W13 + b2).relu();
    Value W22 = (w22 * W11 + w22 * W12 + w22 * W13 + b2).relu();

    Value W31 = (w31 * W21 + w31 * W22).tanh();
    Value W32 = (w32 * W21 + w32 * W22).tanh();

    return W31 + W32 + b3;
  };

  scalar_t const eps = 1e-2;
  for (int its = 0; its < 100000; its++) {
    Value loss{0};
    for (int x : {0, 1}) {
      for (int y : {0, 1}) {
        Value output = forward(x, y);
        Value error = ((x ^ y) - output);
        loss = loss + error * error;
      }
    }

    loss = loss / 4;
    if (its % 1000 == 0)
      std::cout << "After " << its << " epochs, loss is " << loss.get_data() << std::endl;

    // zero_grad included
    loss.backward();

    for (auto const &param : {w11, w12, w13, w21, w22, w31, w32, b1, b2, b3}) 
      param.set_data(param.get_data() - eps * param.get_grad());
  }

  for (int x : {0, 1}) {
    for (int y : {0, 1}) {
      Value output = forward(x, y);

      std::cout << "For input: " << x << " " << y << ", output is " << output.get_data() << std::endl;
    }
  }
}
