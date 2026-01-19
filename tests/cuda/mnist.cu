#include "../mnist/mnist_loader.hpp"
#include "launcher.hpp"
#include <string>

using Scalar = cuda_mlp::CudaScalar;

int main() {
  // Load a small MNIST subset for quick tests.
  auto train_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/train-images.idx3-ubyte", 10000);
  auto train_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/train-labels.idx1-ubyte", 10000);
  auto test_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/t10k-images.idx3-ubyte", 1000);
  auto test_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/t10k-labels.idx1-ubyte", 1000);

  cuda_mlp::Dataset<Scalar> data{train_x, train_y, test_x, test_y};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  // Minimal MLP for MNIST.
  runner.buildNetwork({{784, 128, cuda_mlp::ActivationType::Tanh}, {128, 10, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs;

  const int lbfgs_iters = 200;
  const float lbfgs_tol = 1e-4f;
  const size_t lbfgs_memories[] = {10, 20};
  for (size_t mem : lbfgs_memories) {
    runs.push_back(cuda_mlp::RunConfig{.name = "LBFGS_m" + std::to_string(mem) + "_it" + std::to_string(lbfgs_iters),
        .optimizer = cuda_mlp::OptimizerType::LBFGS,
        .max_iters = lbfgs_iters,
        .tolerance = lbfgs_tol,
        .lbfgs_memory = mem,
        .log_interval = 5});
  }

  const int sgd_epochs = 200;
  const int sgd_batch = 128;
  const int sgd_decay_step = 25;
  const float sgd_tol = 1e-4f;
  const float sgd_lrs[] = {0.05f, 0.01f};
  const float sgd_moms[] = {0.0f, 0.9f};
  const float sgd_decays[] = {1.0f, 0.95f};
  for (float lr : sgd_lrs) {
    for (float momentum : sgd_moms) {
      for (float decay : sgd_decays) {
        runs.push_back(cuda_mlp::RunConfig{
            .name = "SGD_lr" + std::to_string(lr) + "_m" + std::to_string(momentum) + "_d" + std::to_string(decay),
            .optimizer = cuda_mlp::OptimizerType::SGD,
            .max_iters = sgd_epochs,
            .tolerance = sgd_tol,
            .batch_size = sgd_batch,
            .gd_lr = lr,
            .gd_momentum = momentum,
            .sgd_decay_rate = decay,
            .sgd_decay_step = sgd_decay_step,
            .log_interval = 10});
      }
    }
  }

  runner.runExperiments(runs, "results_mnist.csv");

  return 0;
}
