#include "../../src/cuda/network.cuh"
#include "../../src/cuda/lbfgs.cuh"
#include "../cuda_report.hpp"
#include "mnist_loader.hpp"
#include <Eigen/Core>
#include <chrono>
#include <iostream>

using Scalar = cuda_mlp::CudaScalar;
using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

int main(int argc, char **argv) {

  int train_size = 5000;
  int test_size = 1000;
  int max_iters = 500;
  Scalar tolerance = 1e-4;

  std::cout << "Loading Training Data..." << std::endl;
  Mat train_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/train-images.idx3-ubyte", train_size);
  Mat train_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/train-labels.idx1-ubyte", train_size);

  std::cout << "Loading Test Data..." << std::endl;
  Mat test_x = MNISTLoader::loadImages<Scalar>("../tests/mnist/t10k-images.idx3-ubyte", test_size);
  Mat test_y = MNISTLoader::loadLabels<Scalar>("../tests/mnist/t10k-labels.idx1-ubyte", test_size);

  cuda_mlp::CublasHandle handle;
  cuda_mlp::CudaNetwork network(handle);
  network.addLayer(784, 128, cuda_mlp::ActivationType::Tanh);
  network.addLayer(128, 10, cuda_mlp::ActivationType::Linear);
  network.bindParams();

  cuda_mlp::DeviceBuffer<Scalar> d_train_x, d_train_y, d_test_x, d_test_y;

  d_train_x.copy_from_host(train_x.data(), train_x.size());
  d_train_y.copy_from_host(train_y.data(), train_y.size());
  d_test_x.copy_from_host(test_x.data(), test_x.size());
  d_test_y.copy_from_host(test_y.data(), test_y.size());

  std::cout << "\nStarting Training..." << std::endl;
  auto train_start = std::chrono::steady_clock::now();

  cuda_mlp::CudaLBFGS solver(handle);
  solver.setMaxIterations(max_iters);
  solver.setTolerance(tolerance);
  const int params_size = static_cast<int>(network.params_size());
  auto loss_grad = [&](const Scalar *params, Scalar *grad, const Scalar *input, const Scalar *target, int batch) -> Scalar {
    (void)params;
    Scalar loss = network.compute_loss_and_grad(input, target, batch);
    cuda_mlp::device_copy(grad, network.grads_data(), params_size);
    return loss;
  };
  solver.solve(params_size, network.params_data(), d_train_x.data(), d_train_y.data(), train_size, loss_grad);

  auto train_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> train_elapsed = train_end - train_start;
  std::cout << "\nTraining time: " << train_elapsed.count() << " s" << std::endl;

  cuda_tests::print_report("TRAINING SET RESULTS:", network, d_train_x, train_y, train_size);
  cuda_tests::print_report("TEST SET RESULTS:", network, d_test_x, test_y, test_size);

  return 0;
}
