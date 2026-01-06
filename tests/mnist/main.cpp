#include "mnist_loader.hpp"
#include "../../src/network.hpp"
#include "../../src/lbfgs.hpp"
#include "../../src/common.hpp"
#include "../../src/s_lbfgs.hpp"


using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
  checkParallelism();
  
    int train_size = 5000;
    int test_size = 1000;

    std::cout << "Loading Training Data..." << std::endl;
    Mat train_x = MNISTLoader::loadImages("../tests/mnist/train-images.idx3-ubyte", train_size);
    Mat train_y = MNISTLoader::loadLabels("../tests/mnist/train-labels.idx1-ubyte", train_size);

    std::cout << "Loading Test Data..." << std::endl;
    Mat test_x = MNISTLoader::loadImages("../tests/mnist/t10k-images.idx3-ubyte", test_size);
    Mat test_y = MNISTLoader::loadLabels("../tests/mnist/t10k-labels.idx1-ubyte", test_size);

    Network network;
    network.addLayer<784, 128, Tanh>();
    network.addLayer<128, 10, Linear>();
    network.bindParams();


    /*
    
    std::shared_ptr<SLBFGS<Vec, Mat>> solver = std::make_shared<SLBFGS<Vec, Mat>>();
    solver->setMaxIterations(500);
    solver->setTolerance(1.e-4);
    
    int b=20;      //gradient minibatch size
    int b_H=10*b;    //Hessian minibatch siz
    int m=train_size/b;    //number of minibatches
    int M_param=10; //memory parameter
    int L=10;       //number of epochs between Hessian updates
    double step_size=0.05; //SGD step size
    
    std::cout << "\nStarting Training..." << std::endl;
    
    
    network.train_stochastic(train_x, train_y, solver, m, M_param, L, b, b_H, step_size, true, 1000);
    */


    std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<LBFGS<Vec, Mat>>();
    solver->setMaxIterations(500);
    solver->setTolerance(1.e-4);

    std::cout << "\nStarting Training..." << std::endl;
    network.train(train_x, train_y, solver);


    std::cout << "\nTRAINING SET RESULTS:" << std::endl;
    network.test(train_x, train_y);

    std::cout << "\nTEST SET RESULTS:" << std::endl;
    network.test(test_x, test_y);

    return 0;
}
