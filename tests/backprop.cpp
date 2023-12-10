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

TEST(Backprop, AddSelf) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a + a + a + b + b + a + b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a + t_a + t_a + t_b + t_b + t_a + t_b;

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

TEST(Backprop, AddAssValue) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a + b;
  ab += a;
  ab += ab;
  ab += b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a + t_b;
  t_ab += t_a;
  t_ab += t_ab;
  t_ab += t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, AddAssScalar) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a + b;
  ab += 1;
  ab += 2;
  ab += 3;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a + t_b;
  t_ab += 1;
  t_ab += 2;
  t_ab += 3;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Sub) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a - b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a - t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, SubSelf) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a - a - a - b - b - a - b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a - t_a - t_a - t_b - t_b - t_a - t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Sub2) {
  Value a{2.1};
  Value b{3.1};
  Value ab = a - b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a - t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, SubAssValue) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a - b;
  ab -= a;
  ab -= ab;
  ab -= b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a - t_b;
  t_ab -= t_a;
  t_ab -= t_ab;
  t_ab -= t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, SubAssScalar) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a - b;
  ab -= 1;
  ab -= 2;
  ab -= 3;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a - t_b;
  t_ab -= 1;
  t_ab -= 2;
  t_ab -= 3;

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


TEST(Backprop, MultSelf) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a * a * a * b * b * a * b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a * t_a * t_a * t_b * t_b * t_a * t_b;

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

TEST(Backprop, MultAssValue) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a * b;
  ab *= a;
  // ab *= ab;
  ab *= b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a * t_b;
  t_ab *= t_a;
  // t_ab *= t_ab;
  t_ab *= t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, MultAssScalar) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a * b;
  ab *= 1;
  ab *= 2;
  ab *= 3;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a * t_b;
  t_ab *= 1;
  t_ab *= 2;
  t_ab *= 3;

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

TEST(Backprop, DivSelf) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a / a / b / b / a / b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a / t_a / t_b / t_b / t_a / t_b;

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

TEST(Backprop, DivAssValue) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a / b;
  ab /= a;
  // ab /= ab;
  ab /= b;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a / t_b;
  t_ab /= t_a;
  // t_ab /= t_ab;
  t_ab /= t_b;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, DivAssScalar) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a / b;
  ab /= 1;
  ab /= 2;
  ab /= 3;

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a / t_b;
  t_ab /= 1;
  t_ab /= 2;
  t_ab /= 3;

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Sin) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a.sin() * b.sin()).sin();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a.sin() * t_b.sin()).sin();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Sin2) {
  Value a{2.213213};
  Value b{3.137127};
  Value ab = (b * (a.sin().sin() + 5) * b.sin()).sin().sin();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_b * (t_a.sin().sin() + 5) * t_b.sin()).sin().sin();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Cos) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a.cos() * b.cos()).cos();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a.cos() * t_b.cos()).cos();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Cos2) {
  Value a{2.213213};
  Value b{3.137127};
  Value ab = (b * (a.cos().cos() + 5) * b.cos()).cos().cos();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_b * (t_a.cos().cos() + 5) * t_b.cos()).cos().cos();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Exp) {
  Value a{2.0};
  Value b{3.0};
  Value ab = a.exp() * b.exp();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_a.exp() * t_b.exp();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Exp2) {
  Value a{0.813213};
  Value b{1.137127};
  Value ab = b * a.exp().exp() * b.exp();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = t_b * t_a.exp().exp() * t_b.exp();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Tanh) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a.tanh() * b.tanh()).tanh();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a.tanh() * t_b.tanh()).tanh();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Tanh2) {
  Value a{2.213213};
  Value b{3.137127};
  Value ab = (b * (a.tanh().tanh() + 5) * b.tanh()).tanh().tanh();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_b * (t_a.tanh().tanh() + 5) * t_b.tanh()).tanh().tanh();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Relu) {
  Value a{2.0};
  Value b{3.0};
  Value ab = (a.relu() * b.relu()).relu();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_a.relu() * t_b.relu()).relu();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Relu2) {
  Value a{2.213213};
  Value b{3.137127};
  Value ab = (b * (a.relu().relu() + 5) * b.relu()).relu().relu();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_b * (t_a.relu().relu() + 5) * t_b.relu()).relu().relu();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}

TEST(Backprop, Relu3) {
  Value a{2.213213};
  Value b{3.137127};
  Value ab = (b * (a.relu().relu() - 5) * (b - 5).relu()).relu().relu();

  torch::Tensor t_a = torch::tensor(a.get_data(), torch::requires_grad());
  torch::Tensor t_b = torch::tensor(b.get_data(), torch::requires_grad());
  auto t_ab = (t_b * (t_a.relu().relu() - 5) * (t_b - 5).relu()).relu().relu();

  ab.backward();
  t_ab.backward();
    
  EXPECT_NEAR(t_ab.data().item<scalar_t>(), ab.get_data(), eps);
  EXPECT_NEAR(t_a.grad().item<scalar_t>(), a.get_grad(), eps);
  EXPECT_NEAR(t_b.grad().item<scalar_t>(), b.get_grad(), eps);
}
