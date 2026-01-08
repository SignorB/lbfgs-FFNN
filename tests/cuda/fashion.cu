#include "../mnist/mnist_loader.hpp"
#include "launcher.hpp"
#include <iostream>
#include <vector>

using Scalar = cuda_mlp::CudaScalar;

int main() {
  int n_train = 60000;
  int n_test = 10000;

  std::cout << "Loading Fashion-MNIST..." << std::endl;
  auto train_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/train-images-idx3-ubyte", n_train);
  auto test_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/t10k-images-idx3-ubyte", n_test);

  cuda_mlp::Dataset<Scalar> data{train_x, train_x, test_x, test_x};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  runner.buildNetwork({{784, 256, cuda_mlp::ActivationType::Tanh},
      {256, 64, cuda_mlp::ActivationType::Tanh},
      {64, 256, cuda_mlp::ActivationType::Tanh},
      {256, 784, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs;

  runs.push_back({.name = "GD_LR_0.01",
      .optimizer = cuda_mlp::OptimizerType::GD,
      .max_iters = 1000,
      .gd_lr = 0.01f,
      .log_interval = 10});
  runs.push_back({.name = "GD_LR_0.05",
      .optimizer = cuda_mlp::OptimizerType::GD,
      .max_iters = 1000,
      .gd_lr = 0.05f,
      .log_interval = 10});
  runs.push_back(
      {.name = "GD_LR_0.1", .optimizer = cuda_mlp::OptimizerType::GD, .max_iters = 1000, .gd_lr = 0.1f, .log_interval = 10});
  runs.push_back(
      {.name = "GD_LR_0.2", .optimizer = cuda_mlp::OptimizerType::GD, .max_iters = 1000, .gd_lr = 0.2f, .log_interval = 10});
  runs.push_back({.name = "LBFGS_m05",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 5,
      .log_interval = 5});
  runs.push_back({.name = "LBFGS_m10",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 10,
      .log_interval = 5});
  runs.push_back({.name = "LBFGS_m20",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 20,
      .log_interval = 5});
  runs.push_back({.name = "LBFGS_m50",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 50,
      .log_interval = 5});
  runs.push_back({.name = "LBFGS_m100",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 100,
      .log_interval = 5});
  runs.push_back({.name = "LBFGS_m200",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 500,
      .lbfgs_memory = 200,
      .log_interval = 5});
  runs.push_back(
      {.name = "GD_Long", .optimizer = cuda_mlp::OptimizerType::GD, .max_iters = 3000, .gd_lr = 0.05f, .log_interval = 50});
  runs.push_back({.name = "LBFGS_Long",
      .optimizer = cuda_mlp::OptimizerType::LBFGS,
      .max_iters = 1000,
      .lbfgs_memory = 20,
      .log_interval = 10});
  std::cout << "Starting Benchmark Suite..." << std::endl;
  runner.runExperiments(runs, "final_benchmark_summary.csv");

  return 0;
}