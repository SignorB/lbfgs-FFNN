#include "../mnist/mnist_loader.hpp"
#include "launcher.hpp"

using Scalar = cuda_mlp::CudaScalar;

int main() {
  int n_train = 60000;
  int n_test = 10000;

  auto train_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/train-images-idx3-ubyte", n_train);
  auto test_x = MNISTLoader::loadImages<Scalar>("../tests/fashion-mnist/t10k-images-idx3-ubyte", n_test);

  cuda_mlp::Dataset<Scalar> data{train_x, train_x, test_x, test_x};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  runner.buildNetwork({{784, 256, cuda_mlp::ActivationType::Tanh},
      {256, 64, cuda_mlp::ActivationType::Tanh},
      {64, 256, cuda_mlp::ActivationType::Tanh},
      {256, 784, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs = {
      cuda_mlp::RunConfig{
          .name = "LBFGS_m10",
          .optimizer = cuda_mlp::OptimizerType::LBFGS,
          .max_iters = 1000,
          .tolerance = 1e-4f,
          .lbfgs_memory = 10,
      },
      cuda_mlp::RunConfig{
          .name = "LBFGS_m100",
          .optimizer = cuda_mlp::OptimizerType::LBFGS,
          .max_iters = 1000,
          .tolerance = 1e-4f,
          .lbfgs_memory = 100,
      },
      cuda_mlp::RunConfig{
          .name = "GD Baseline",
          .optimizer = cuda_mlp::OptimizerType::GD,
          .max_iters = 300,
          .gd_lr = 0.01f,
          .gd_momentum = 0.9f,
      },
  };

  runner.runExperiments(runs, "results_fmnist_ae.csv");

  return 0;
}