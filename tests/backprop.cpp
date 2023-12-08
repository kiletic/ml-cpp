#include <gtest/gtest.h>

#include <torch/torch.h>
#include "value.h"

double const eps = 5e-4;

TEST(Backprop, Add) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a + b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a + t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Add2) {
  Value a{2.1};
  Value b{3.1};
  Value ab = a + b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a + t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Mult) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a * b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a * t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Mult2) {
  Value a{2.1};
  Value b{3.1};
  Value ab = a * b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a * t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, AddMult) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a + b) * a;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a + t_b) * t_a;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, MultAdd) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a * b) + a;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a * t_b) + t_a;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, MultAdd2) {
  Value a{2.1};
  Value b{3.1};
  Value ab = (a * b) + a;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a * t_b) + t_a;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, MultAdd3) {
  Value a{2.323};
  Value b{3.834};
  Value ab = ((a + 2 * b) * (a * b) + a) + (b * a) * b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = ((t_a + t_b * 2) * (t_a * t_b) + t_a) + (t_b * t_a) * t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Div) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a / b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a / t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Div2) {
  Value a{2.2331};
  Value b{3.4472};
  Value ab = ((a + b) * b / a) / (a * b / a * 5 * b);

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = ((t_a + t_b) * t_b / t_a) / (t_a * t_b / t_a * 5 * t_b);

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, DivScalar) {
  Value a{2.2331};
  Value b{3.4472};
  Value ab = (((a / 2) + (3 / b)) * b / a) / ((a / 7.5) * b / a * 5 * (b / 8.6));

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (((t_a / 2) + (3 / t_b)) * t_b / t_a) / ((t_a / 7.5) * t_b / t_a * 5 * (t_b / 8.6));

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}
