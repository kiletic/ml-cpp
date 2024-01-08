#include <iomanip>
#include <iostream>

#include <torch/torch.h>

struct Net : torch::nn::Module {
  torch::nn::Linear fc1, fc2;

  Net() : 
    fc1(register_module("linear1", torch::nn::Linear(2, 2))),
    fc2(register_module("linear2", torch::nn::Linear(2, 1)))
  {}

  torch::Tensor forward(torch::Tensor x) {
    x = torch::tanh(fc1(x));
    return fc2(x);
  }
};

int main() {
  std::cout << std::setprecision(15) << std::fixed;

  Net model;
  auto const params = model.parameters();
  
  double const eps = 1e-2;
  for (int its = 0; its < 100000; its++) {
    torch::Tensor loss = torch::zeros(1, torch::dtype(torch::kFloat32)); 
    for (int x : {0, 1}) {
      for (int y : {0, 1}) {
        torch::Tensor in = torch::tensor({x, y}, {torch::kFloat32});
        torch::Tensor output_tensor = model.forward(in);
        auto error = output_tensor - (x ^ y);
        loss += error * error;
      }
    }

    loss = loss / 4;
    if (its % 1000 == 0)
      std::cout << "After " << its << " epochs, loss is " << loss.item<double>() << std::endl;

    loss.backward();

    for (auto const &param : params) 
      param.data() -= eps * param.grad();

    for (auto const &param : params) 
      param.grad().zero_();
  }

  for (int x : {0, 1}) {
    for (int y : {0, 1}) {
      torch::Tensor in = torch::tensor({x, y}, {torch::kFloat32});
      torch::Tensor output_tensor = model.forward(in);
      auto output = output_tensor.item<double>();

      std::cout << "For input: " << x << " " << y << ", output is " << output << std::endl;
    }
  }
}
