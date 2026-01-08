#include "../mnist/mnist_loader.hpp"
#include "launcher.hpp"

using Scalar = cuda_mlp::CudaScalar;

int main() {
  auto train_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/train-images.idx3-ubyte", 5000);
  auto train_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/train-labels.idx1-ubyte", 5000);
  auto test_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/t10k-images.idx3-ubyte", 1000);
  auto test_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/t10k-labels.idx1-ubyte", 1000);

  cuda_mlp::Dataset<Scalar> data{train_x, train_y, test_x, test_y};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  runner.buildNetwork({{784, 128, cuda_mlp::ActivationType::Tanh}, {128, 10, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs = {
      cuda_mlp::RunConfig{.name = "LBFGS_Standard",
          .optimizer = cuda_mlp::OptimizerType::LBFGS,
          .max_iters = 200,
          .tolerance = 1e-4f,
          .lbfgs_memory = 20},
      cuda_mlp::RunConfig{.name = "GD_Momentum",
          .optimizer = cuda_mlp::OptimizerType::GD,
          .max_iters = 50000,
          .tolerance = 1e-2f,
          .gd_lr = 0.f,
          .gd_momentum = 0.9f},
  };

  runner.runExperiments(runs, "results_mnist.csv");

  return 0;
}