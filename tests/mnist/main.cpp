#include "mnist_loader.hpp"
#include "../../src/network.hpp"
#include "../../src/lbfgs.hpp"
#include "../../src/common.hpp"
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;


int main() {
  checkParallelism();
    int batch_size = 500;
    
    std::cout << "Loading MNIST Data..." << std::endl;
    Eigen::MatrixXd training_inputs = MNISTLoader::loadImages("../tests/mnist/train-images.idx3-ubyte", batch_size);
    Eigen::MatrixXd training_targets = MNISTLoader::loadLabels("../tests/mnist/train-labels.idx1-ubyte", batch_size);

    Network network;
    network.addLayer<784, 128, Tanh>(); 
    network.addLayer<128, 10, Linear>(); 
    
    network.bindParams();

    std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<LBFGS<Vec, Mat>>();
    solver->setMaxIterations(200);
    solver->setTolerance(1.e-5);

    std::cout << "Starting Training..." << std::endl;
    
    network.train(training_inputs, training_targets, solver);

    return 0;
}
