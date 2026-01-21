#include "../../src/common.hpp"
#include "../../src/unified_launcher.hpp"
#include "../mnist/mnist_loader.hpp"
#include <iostream>
#include <omp.h>

using Backend = CudaBackend;

int main() {
  checkParallelism();
  UnifiedLauncher<Backend> launcher;

  std::cout << "Building Network..." << std::endl;
  launcher.addLayer<784, 128, cpu_mlp::ReLU>();
  launcher.addLayer<128, 10, cpu_mlp::Linear>();
  launcher.buildNetwork();

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

  launcher.setData(dataset);

  {
    UnifiedConfig config;
    config.name = "FASHION_MNIST_GD";
    config.max_iters = 500;
    config.tolerance = 1e-2;
    config.learning_rate = 0.02;
    config.momentum = 0.9;
    config.log_interval = 1;

    std::cout << "Running GD..." << std::endl;
    UnifiedGD<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  {
    UnifiedConfig config;
    config.name = "FASHION_MNIST_SGD";
    config.max_iters = 1000;
    config.tolerance = 1e-2;
    config.learning_rate = 0.01;
    config.batch_size = 128;
    config.log_interval = 5;
    config.lr_decay = 0.70;
    config.lr_decay_rate = 40;

    std::cout << "Running SGD..." << std::endl;
    UnifiedSGD<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  {
    UnifiedConfig config;
    config.name = "FASHION_LBFGS_m100";
    config.max_iters = 1000;
    config.tolerance = 1e-2;
    config.m_param = 100;
    config.log_interval = 1;

    std::cout << "Running LBFGS..." << std::endl;
    UnifiedLBFGS<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  {
    UnifiedConfig config;
    config.name = "FASHION_MNIST_LBFGS_m10";
    config.max_iters = 1000;
    config.tolerance = 1e-2;
    config.m_param = 10;
    config.log_interval = 1;

    std::cout << "Running LBFGS..." << std::endl;
    UnifiedLBFGS<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  return 0;
}
