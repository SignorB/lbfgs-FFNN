#include "bfgs.hpp"
#include "layer.hpp"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <iostream>
#include <memory>

using namespace autodiff;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

int main() {
  Network network;
  network.addLayer(DenseLayer(8, 12));
  network.addLayer(DenseLayer(12, 4, true));

  network.bindParams();

  VectorXvar v(8);
  v << 0.5, -0.2, 1.0, 0.1, -0.5, 0.8, 0.3, -0.1;

  std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<BFGS<Vec, Mat>>();
  solver->setMaxIterations(4000);
  solver->setTolerance(1.e-14);
  int n = network.getSize();

  Mat m(n, n);
  for (int i = 0; i < n; ++i)
    m(i, i) = 1;
  solver->setInitialHessian(m);

  network.train(solver);

  std::cout << network.forward(v) << std::endl;
}
