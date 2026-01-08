#include "../rand/bin_loader.hpp"
#include "launcher.hpp"

using Scalar = cuda_mlp::CudaScalar;

int main() {

  auto train_x = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/train_x.bin", 300000);
  auto train_y = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/train_y.bin", 300000);
  auto test_x = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/test_x.bin", 50000);
  auto test_y = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/test_y.bin", 50000);

  cuda_mlp::Dataset<Scalar> data{train_x, train_y, test_x, test_y};

  cuda_mlp::ExperimentRunner<Scalar> runner;
  runner.setData(data);

  runner.buildNetwork({{2048, 1024, cuda_mlp::ActivationType::Tanh}, {1024, 10, cuda_mlp::ActivationType::Linear}});

  std::vector<cuda_mlp::RunConfig> runs = {
      // cuda_mlp::RunConfig{.name = "LBFGS",
      //     .optimizer = cuda_mlp::OptimizerType::LBFGS,
      //     .max_iters = 200,
      //     .tolerance = 1e-2f,
      //     .lbfgs_memory = 20},
      cuda_mlp::RunConfig{.name = "GD",
          .optimizer = cuda_mlp::OptimizerType::GD,
          .max_iters = 200,
          .tolerance = 1e-2f,
          .gd_lr = 0.1f,
          .gd_momentum = 0.9f},
  };
  runner.runExperiments(runs, "results_synthetic.csv");

  return 0;
}