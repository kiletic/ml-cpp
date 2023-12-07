#include <gtest/gtest.h>

#include <torch/torch.h>
#include "value.h"


TEST(Backprop, Addition) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a + b;

  torch::Tensor t_a = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor t_b = torch::tensor(3.0, torch::requires_grad());
  auto t_ab = t_a + t_b;

  ab.backward();
  t_ab.backward();
    
  ASSERT_EQ(t_a.grad().item<scalar_t>(), a.get_grad());
  ASSERT_EQ(t_b.grad().item<scalar_t>(), b.get_grad());
}
