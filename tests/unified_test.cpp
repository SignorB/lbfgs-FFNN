#include "../src/launcher_unified.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // 1. Create Launcher for CPU
    UnifiedLauncher<CpuBackend> launcher;

    // 2. Build Network (Static Architecture)
    // 2 inputs, 5 hidden, 1 output
    launcher.addLayer<2, 5, Sigmoid>();
    launcher.addLayer<5, 1, Linear>();
    launcher.buildNetwork();

    std::cout << "Network built successfully on CPU backend." << std::endl;

    // 3. Create Dummy Data (XOR-like)
    UnifiedDataset data;
    data.train_x = Eigen::MatrixXd::Random(2, 10); // 10 samples
    data.train_y = Eigen::MatrixXd::Random(1, 10);
    data.test_x = Eigen::MatrixXd::Random(2, 5);
    data.test_y = Eigen::MatrixXd::Random(1, 5);

    launcher.setData(data);

    // 4. Configure Run
    UnifiedConfig config;
    config.name = "TestCPU";
    config.max_iters = 5; // Short run
    config.learning_rate = 0.01;

    // 5. Train
    UnifiedGD<CpuBackend> optimizer;
    launcher.train(optimizer, config);

    std::cout << "Training finished." << std::endl;

    return 0;
}
