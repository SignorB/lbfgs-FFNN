#include "mnist_loader.hpp"
#include "../../src/network.hpp"
#include "../../src/lbfgs.hpp"

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;


int main() {
    std::cout << "Loading MNIST..." << std::endl;
    auto training_inputs = MNISTLoader::loadImages("../tests/mnist/train-images.idx3-ubyte");
    auto training_targets = MNISTLoader::loadLabels("../tests/mnist/train-labels.idx1-ubyte");

    Network network;
    network.addLayer<784, 128, ReLU>(); 
    network.addLayer<128, 10, Linear>(); 
    
    network.bindParams();

    std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<LBFGS<Vec, Mat>>();
    solver->setMaxIterations(4000);
    solver->setTolerance(1.e-14);

    
    std::vector<std::vector<double>> batch_in(training_inputs.begin(), training_inputs.begin() + 1000);
    std::vector<std::vector<double>> batch_out(training_targets.begin(), training_targets.begin() + 1000);

    network.train(batch_in, batch_out, solver);
}
