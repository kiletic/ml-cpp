#include "loss.h"

namespace Loss {

Value binary_cross_entropy(Value const &y, int label) {
  return -label * y.log() - (1 - label) * (1 - y).log(); 
}

Value squared_error(Value const &x, scalar_t y) {
  Value error = x - y;
  return error * error;
}

}
