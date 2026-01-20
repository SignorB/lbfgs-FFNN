#include "../src/launcher_unified.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    UnifiedLauncher<CpuBackend> launcher;

    // 2 inputs, 5 hidden, 1 output
    launcher.addLayer<2, 5, Tanh>(); 
    launcher.addLayer<5, 1, Linear>();
    launcher.buildNetwork();

    std::cout << "Network built successfully on CPU backend (SGD Test)." << std::endl;

    // XOR-like data (noisy)
    int n_samples = 1000;
    UnifiedDataset data;
    data.train_x = Eigen::MatrixXd::Random(2, n_samples);
    data.train_y = Eigen::MatrixXd::Zero(1, n_samples);
    // Simple rule: y = x0 * x1
    for(int i=0; i<n_samples; ++i) {
        data.train_y(0, i) = data.train_x(0, i) * data.train_x(1, i);
    }
    
    data.test_x = Eigen::MatrixXd::Random(2, 100);
    data.test_y = Eigen::MatrixXd::Zero(1, 100);
    for(int i=0; i<100; ++i) {
        data.test_y(0, i) = data.test_x(0, i) * data.test_x(1, i);
    }

    launcher.setData(data);

    UnifiedConfig config;
    config.name = "TestSGD_Batch";
    config.max_iters = 100; // Epochs
    config.learning_rate = 0.1;
    config.batch_size = 32;
    config.log_interval = 20;

    // Train
    UnifiedSGD<CpuBackend> optimizer;
    launcher.train(optimizer, config);

    std::cout << "SGD Training (Batch Optimized) finished." << std::endl;

    return 0;
}
