#include "../../src/common.hpp"
#include "../../src/unified_launcher.hpp"
#include "../mnist/mnist_loader.hpp"
#include <iostream>
#include <string>
#include <vector>

using Backend = CpuBackend;

int main() {
  checkParallelism();

  int train_size = 60000;
  int test_size = 10000;

  std::cout << "Loading Fashion-MNIST..." << std::endl;
  Eigen::MatrixXd train_x =
      MNISTLoader::loadImages("../tests/fashion-mnist/FashionMNIST/raw/train-images-idx3-ubyte", train_size);
  Eigen::MatrixXd train_y =
      MNISTLoader::loadLabels("../tests/fashion-mnist/FashionMNIST/raw/train-labels-idx1-ubyte", train_size);
  Eigen::MatrixXd test_x =
      MNISTLoader::loadImages("../tests/fashion-mnist/FashionMNIST/raw/t10k-images-idx3-ubyte", test_size);
  Eigen::MatrixXd test_y =
      MNISTLoader::loadLabels("../tests/fashion-mnist/FashionMNIST/raw/t10k-labels-idx1-ubyte", test_size);

  UnifiedDataset dataset;
  dataset.train_x = train_x;
  dataset.train_y = train_y;
  dataset.test_x = test_x;
  dataset.test_y = test_y;

  auto run_experiment = [&](const UnifiedConfig &config, auto optimizer) {
    UnifiedLauncher<Backend> launcher;
    launcher.addLayer<784, 128, Tanh>();
    launcher.addLayer<128, 10, Linear>();
    launcher.buildNetwork();
    launcher.setData(dataset);

    launcher.train(optimizer, config);
    launcher.test();
  };

  const double tol = 2e-1;

  const int lbfgs_iters = 1000;
  const int lbfgs_memories[] = {5, 10, 20, 50};
  for (int mem : lbfgs_memories) {
    UnifiedConfig config;
    config.name = "LBFGS_m" + std::to_string(mem) + "_it" + std::to_string(lbfgs_iters);
    config.max_iters = lbfgs_iters;
    config.tolerance = tol;
    config.m_param = mem;
    config.log_interval = 5;

    std::cout << "Running " << config.name << "..." << std::endl;
    UnifiedLBFGS<Backend> optimizer;
    run_experiment(config, optimizer);
  }

  const int sgd_epochs = 1200;
  const int sgd_batch = 256;
  const double sgd_mom = 0.9;
  const double sgd_lrs[] = {0.05, 0.01};
  for (double lr : sgd_lrs) {
    UnifiedConfig config;
    config.name = "SGD_lr" + std::to_string(lr);
    config.max_iters = sgd_epochs;
    config.tolerance = tol;
    config.batch_size = sgd_batch;
    config.learning_rate = lr;
    config.momentum = sgd_mom;
    config.log_interval = 10;

    std::cout << "Running " << config.name << "..." << std::endl;
    UnifiedSGD<Backend> optimizer;
    run_experiment(config, optimizer);
  }

  const int gd_iters = 5000;
  const double gd_lrs[] = {0.05, 0.01};
  const double gd_moms[] = {0.0, 0.9};
  for (double lr : gd_lrs) {
    for (double momentum : gd_moms) {
      UnifiedConfig config;
      config.name = "GD_lr" + std::to_string(lr) + "_m" + std::to_string(momentum);
      config.max_iters = gd_iters;
      config.tolerance = tol;
      config.learning_rate = lr;
      config.momentum = momentum;
      config.log_interval = 5;

      std::cout << "Running " << config.name << "..." << std::endl;
      UnifiedGD<Backend> optimizer;
      run_experiment(config, optimizer);
    }
  }

  return 0;
}
