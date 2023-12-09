#include <memory>

#include "value.h"
#include "layer.h"

struct NeuralNet {
  std::vector<std::unique_ptr<Layer>> network;

  NeuralNet& operator<<(Layer*);
};
