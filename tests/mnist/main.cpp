#include "../../src/unified_launcher.hpp"
#include "../../src/common.hpp"
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
    launcher.buildNetwork();

    int train_size = 5000;
    int test_size = 1000;

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

    UnifiedConfig config;
    config.name = "MNIST_Unified_GD";
    config.max_iters = 30;
    config.tolerance = 1e-4;
    config.learning_rate = 0.05; // Normalized gradients allow higher LR, but 1.0 was too edge-case. Trying 0.5.
    config.log_interval = 1;

    // Strategy Pattern: Instantiate Optimizer
    std::cout << "Instantiating Optimizer Strategy (SGD)..." << std::endl;
    UnifiedSGD<Backend> optimizer; 
    
    std::cout << "Starting Training..." << std::endl;
    launcher.train(optimizer, config);

    std::cout << "\n>>> Evaluating on Test Set..." << std::endl;
    launcher.test();

    return 0;
}
