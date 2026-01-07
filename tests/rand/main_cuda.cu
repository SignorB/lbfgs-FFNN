#include "../../src/cuda/network.cuh"
#include "../../src/cuda/optimizer_lbfgs.cuh"
#include "../../src/cuda/optimizer_sgd.cuh"
#include "../cuda_report.hpp"
#include "bin_loader.hpp"
#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <string>

using Scalar = cuda_mlp::CudaScalar;
using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

int main(int argc, char **argv) {
  int train_size = 150000;
  int test_size = 10000;
  int max_iters = 400;
  Scalar tolerance = 1e-3f;
  int hidden_dim = 256;

  std::cout << "Loading Training Data..." << std::endl;
  Mat train_x = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/train_x.bin", train_size);
  Mat train_y = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/train_y.bin", train_size);

  std::cout << "Loading Test Data..." << std::endl;
  Mat test_x = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/test_x.bin", test_size);
  Mat test_y = RandLoader::loadMatrix<Scalar>("../tests/rand/data_bench/test_y.bin", test_size);

  train_size = static_cast<int>(train_x.cols());
  test_size = static_cast<int>(test_x.cols());
  int input_dim = static_cast<int>(train_x.rows());
  int output_dim = static_cast<int>(train_y.rows());

  std::cout << "Input dim: " << input_dim << ", Output dim: " << output_dim << std::endl;
  std::cout << "Train batch: " << train_size << ", Test batch: " << test_size << std::endl;
  std::cout << "Hidden dim: " << hidden_dim << std::endl;

  cuda_mlp::CublasHandle handle;
  cuda_mlp::CudaNetwork network(handle);
  network.addLayer(input_dim, hidden_dim, cuda_mlp::ActivationType::Tanh);
  network.addLayer(hidden_dim, output_dim, cuda_mlp::ActivationType::Linear);
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
  solver.solve(network, d_train_x.data(), d_train_y.data(), train_size);
  auto train_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> train_elapsed = train_end - train_start;

  std::cout << "\nTraining time: " << train_elapsed.count() << " s" << std::endl;
  cuda_tests::print_report("TRAINING SET RESULTS:", network, d_train_x, train_y, train_size);
  cuda_tests::print_report("TEST SET RESULTS:", network, d_test_x, test_y, test_size);

  return 0;
}
