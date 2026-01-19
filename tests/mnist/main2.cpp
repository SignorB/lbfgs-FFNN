#include "../../src/launcher_unified.hpp"
#include "../../src/cuda/cublas_handle.cuh"
#include "../../src/common.hpp"
#include "mnist_loader.hpp"
#include <iostream>
#include <omp.h>

using Backend = CpuBackend; 

int main() {
  Eigen::setNbThreads(4);
  checkParallelism();
  UnifiedLauncher<Backend> launcher;

    std::cout << "Building Network..." << std::endl;
    launcher.addLayer<784, 128, Tanh>();
    launcher.addLayer<128, 10, Linear>();
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
    config.name = "MNIST_Unified_Test";
    config.max_iters = 500;
    config.tolerance = 1.5e-4;
    config.batch_size = 32;   
    config.b_H_param = 96;    
    config.L_param = 20;      
    config.m_param = 10;      
    config.learning_rate = 0.01;
    config.log_interval = 10;

    // Strategy Pattern: Instantiate Optimizer
    std::cout << "Instantiating Optimizer Strategy..." << std::endl;
    UnifiedLBFGS<Backend> optimizer; 
    
    std::cout << "Starting Training..." << std::endl;
    launcher.train(optimizer, config);

    return 0;
}
