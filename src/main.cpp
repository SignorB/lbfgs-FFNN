#include "bfgs.hpp"
#include "network.hpp"
#include <iostream>
#include "s_lbfgs.hpp"
#include "lbfgs.hpp"
#include <memory>
#include <vector>
#include <Eigen/Core>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
    Network network;
    network.addLayer<8, 12, ReLU>();
    network.addLayer<12, 4, Linear>();

    network.bindParams();

    std::vector<double> input = {0.5, -0.2, 1.0, 0.1, -0.5, 0.8, 0.3, -0.1};
    std::vector<double> target = {1.0, 2.0, 3.0, 4.0};

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;

    inputs.push_back(input);
    targets.push_back(target);

    std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<LBFGS<Vec, Mat>>();
    solver->setMaxIterations(4000);
    solver->setTolerance(1.e-14);

    int n = network.getSize();
    Mat m(n, n);
    m.setIdentity();
    solver->setInitialHessian(m);

    network.train(inputs, targets, solver);

    const std::vector<double>& result = network.forward(input);

    for (const auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
