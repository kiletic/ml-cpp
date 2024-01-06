#include "activation_layer.h"
#include "linear_layer.h"
#include "util/csv.h"
#include "neural_network.h"
#include <fstream>

int main() {
  CsvFile train("/home/karlo/Desktop/Karlo/projects/transformer-cpp/examples/kaggle/train.csv"); 

  // Sex, Age
  NeuralNet model;
  // model
  //   .add<LinearLayer>(5, 5)
  //   .add<ActivationLayer<ActivationFunc::tanh>>()
  //   .add<LinearLayer>(5, 50)
  //   .add<ActivationLayer<ActivationFunc::tanh>>()
  //   .add<LinearLayer>(50, 15)
  //   .add<ActivationLayer<ActivationFunc::tanh>>()
  //   .add<LinearLayer>(15, 5)
  //   .add<ActivationLayer<ActivationFunc::tanh>>()
  //   .add<LinearLayer>(5, 1)
  //   .add<ActivationLayer<ActivationFunc::sigmoid>>();

  model
    .add<LinearLayer>(2, 3)
    .add<ActivationLayer<ActivationFunc::tanh>>()
    .add<LinearLayer>(3, 10)
    .add<ActivationLayer<ActivationFunc::relu>>()
    .add<LinearLayer>(10, 2)
    .add<ActivationLayer<ActivationFunc::tanh>>()
    .add<LinearLayer>(2, 1)
    .add<ActivationLayer<ActivationFunc::sigmoid>>();

  auto params = model.get_parameters();
  int const epochs = 100;
  scalar_t const eps = 1e-2;

  for (int epoch = 0; epoch < epochs; epoch++) {
    Value loss = 0;
    int zeros = 0;
    scalar_t avg_y = 0;
    for (int i = 0; i < (int)train.len(); i++) {
      auto raw_x = train.rows[i];
      // scalar_t sex = raw_x["Sex"] == "male";
      // scalar_t age = std::stod(raw_x["Age"] == "" ? "0" : raw_x["Age"]);
      scalar_t fare = std::stod(raw_x["Fare"]);
      scalar_t pclass = std::stod(raw_x["Pclass"]);
      // scalar_t parch = std::stod(raw_x["Parch"]);
      // ValueTensor x({sex, age, fare, pclass, parch});
      // ValueTensor x({sex, age});
      ValueTensor x({fare, pclass});
      auto y = model(x).value();

      if (y.get_data() < 0.5)
        zeros++;
      avg_y += y.get_data();
      // y = P(label(x) = 1)
      int label = std::stoi(raw_x["Survived"]);
      loss += -label * y.log() - (1 - label) * (1 - y).log(); 
    }

    loss /= train.len();
    loss.backward();

    std::cout << "Epoch: " << epoch << ", loss: " << loss.get_data() << std::endl;
    std::cout << "Percentage of zeros: " << (scalar_t)zeros / train.len() << " " << zeros << std::endl;
    std::cout << "Avg_y: " << avg_y / train.len() << std::endl;

    for (auto const &param : params)
      param.set_data(param.get_data() - eps * param.get_grad());
  } 

  CsvFile test("/home/karlo/Desktop/Karlo/projects/transformer-cpp/examples/kaggle/test.csv"); 
  std::ofstream result("/home/karlo/Desktop/Karlo/projects/transformer-cpp/examples/kaggle/submit.csv");
  result << "PassengerId,Survived" << std::endl;

  for (int i = 0; i < (int)test.len(); i++) {
    auto raw_x = test.rows[i];
    // scalar_t sex = raw_x["Sex"] == "male";
    // scalar_t age = std::stod(raw_x["Age"] == "" ? "0" : raw_x["Age"]);
    scalar_t fare = std::stod(raw_x["Fare"] == "" ? "0" : raw_x["Fare"]);
    scalar_t pclass = std::stod(raw_x["Pclass"]);
    // scalar_t parch = std::stod(raw_x["Parch"]);
    // ValueTensor x({sex, age, fare, pclass, parch});
    // ValueTensor x({sex, age});
    ValueTensor x({fare, pclass});
    auto y = model(x).value().get_data();

    result << raw_x["PassengerId"] << "," << (y > 0.5) << std::endl;
  }
}
