#include "../../src/common.hpp"
#include "../../src/unified_launcher.hpp"
#include "mnist_loader.hpp"
#include <iostream>
#include <omp.h>

using Backend = CpuBackend; 

int main() {
  checkParallelism();
  UnifiedLauncher<Backend> launcher;

  std::cout << "Building Network..." << std::endl;
  launcher.addLayer<784, 128, cpu_mlp::Tanh>();
  launcher.addLayer<128, 10, cpu_mlp::Linear>();


  int train_size = 60000;
  int test_size = 10000;
  std::cout << "Loading Training Data..." << std::endl;
  Eigen::MatrixXd train_x = MNISTLoader::loadImages("../tests/mnist/train-images.idx3-ubyte", train_size);
  Eigen::MatrixXd train_y = MNISTLoader::loadLabels("../tests/mnist/train-labels.idx1-ubyte", train_size);

  std::cout << "Loading Test Data..." << std::endl;
  Eigen::MatrixXd test_x = MNISTLoader::loadImages("../tests/mnist/t10k-images.idx3-ubyte", test_size);
  Eigen::MatrixXd test_y = MNISTLoader::loadLabels("../tests/mnist/t10k-labels.idx1-ubyte", test_size);

  UnifiedDataset dataset;
  dataset.train_x = train_x;
  dataset.train_y = train_y;
  dataset.test_x = test_x;
  dataset.test_y = test_y;

  launcher.setData(dataset);

  {
    UnifiedConfig config;
    config.name = "MNIST_Unified_GD";
    config.max_iters = 30;
    config.tolerance = 1e-4;
    config.learning_rate = 0.01;
    config.momentum = 0.9;
    config.log_interval = 5;

    std::cout << "Running GD..." << std::endl;
    UnifiedGD<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  {
    UnifiedConfig config;
    config.name = "MNIST_LBFGS";
    config.max_iters = 1000;
    config.tolerance = 1e-4;
    config.m_param = 20;
    config.log_interval = 2;

    std::cout << "Running LBFGS..." << std::endl;
    UnifiedLBFGS<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  {
    UnifiedConfig config;
    config.name = "MNIST_SGD";
    config.max_iters = 1000;
    config.tolerance = 1e-4;
    config.learning_rate = 0.03;
    config.batch_size = 256;
    config.log_interval = 5;

    std::cout << "Running SGD..." << std::endl;
    UnifiedSGD<Backend> optimizer;
    launcher.train(optimizer, config);
    launcher.test();
  }

  // {
  //   UnifiedConfig config;
  //   config.name = "MNIST_SLBFGS";
  //   config.max_iters = 100;
  //   config.tolerance = 1e-4;
  //   config.learning_rate = 0.02;
  //   config.batch_size = 256;
  //   config.m_param = 10;
  //   config.L_param = 10;
  //   config.b_H_param = 128;
  //   config.log_interval = 5;

  //   std::cout << "Running SLBFGS..." << std::endl;
  //   UnifiedSLBFGS<Backend> optimizer;
  //   launcher.train(optimizer, config);
  //   launcher.test();
  // }

  return 0;
}
