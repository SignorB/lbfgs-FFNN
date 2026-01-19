#include "../mnist/mnist_loader.hpp"
#include "launcher.hpp"
#include <iostream>
#include <string>
#include <vector>

using Scalar = cuda_mlp::CudaScalar;

int main() {
  int n_train = 60000;
  int n_test = 10000;

  // Load Fashion-MNIST dataset into host matrices.
  std::cout << "Loading Fashion-MNIST..." << std::endl;
  auto train_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/FashionMNIST/raw/train-images-idx3-ubyte", n_train);
  auto train_y = MNISTLoader::loadLabels<Scalar>("../tests/fashion-mnist/FashionMNIST/raw/train-labels-idx1-ubyte", n_train);
  auto test_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/FashionMNIST/raw/t10k-images-idx3-ubyte", n_test);
  auto test_y = MNISTLoader::loadLabels<Scalar>("../tests/fashion-mnist/FashionMNIST/raw/t10k-labels-idx1-ubyte", n_test);

  cuda_mlp::Dataset<Scalar> data{train_x, train_y, test_x, test_y};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  // Define a deeper network for Fashion-MNIST.
  runner.buildNetwork({{784, 128, cuda_mlp::ActivationType::Tanh},
      {128, 10, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs;

  const float tol = 2e-1f;

  const int lbfgs_iters = 1000;
  const size_t lbfgs_memories[] = {5, 10, 20, 50};
  for (size_t mem : lbfgs_memories) {
    runs.push_back(cuda_mlp::RunConfig{.name = "LBFGS_m" + std::to_string(mem) + "_it" + std::to_string(lbfgs_iters),
        .optimizer = cuda_mlp::OptimizerType::LBFGS,
        .max_iters = lbfgs_iters,
        .tolerance = tol,
        .lbfgs_memory = mem,
        .log_interval = 5});
  }

  const int sgd_epochs = 1200;
  const int sgd_batch = 256;
  const int sgd_decay_step = 20;
  const float sgd_mom = 0.9f;
  const float sgd_lrs[] = {0.05f, 0.01f};
  const float sgd_decay = 1.0f;
  for (float lr : sgd_lrs) {
    runs.push_back(cuda_mlp::RunConfig{.name = "SGD_lr" + std::to_string(lr),
        .optimizer = cuda_mlp::OptimizerType::SGD,
        .max_iters = sgd_epochs,
        .tolerance = tol,
        .batch_size = sgd_batch,
        .gd_lr = lr,
        .gd_momentum = sgd_mom,
        .sgd_decay_rate = sgd_decay,
        .sgd_decay_step = sgd_decay_step,
        .log_interval = 10});
  }

  const int gd_iters = 5000;
  const float gd_lrs[] = {0.05f, 0.01f};
  const float gd_moms[] = {0.0f, 0.9f};
  for (float lr : gd_lrs) {
    for (float momentum : gd_moms) {
      runs.push_back(cuda_mlp::RunConfig{.name = "GD_lr" + std::to_string(lr) + "_m" + std::to_string(momentum),
          .optimizer = cuda_mlp::OptimizerType::GD,
          .max_iters = gd_iters,
          .tolerance = tol,
          .gd_lr = lr,
          .gd_momentum = momentum,
          .log_interval = 5});
    }
  }
  std::cout << "Starting Benchmark Suite..." << std::endl;
  runner.runExperiments(runs, "final_benchmark_summary.csv");

  return 0;
}
