#include "../../src/cuda/network.cuh"
#include "../../src/cuda/optimizer.cuh"
#include "mnist_loader.hpp"
#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <vector>
#include "../../src/simple_config.hpp"
using Mat = Eigen::MatrixXd;

namespace {

void report_results(cuda_mlp::CudaNetwork &network, const cuda_mlp::DeviceBuffer<double> &inputs, const Mat &targets,
                    int batch) {
  auto start = std::chrono::steady_clock::now();
  network.forward_only(inputs.data(), batch);

  int out_size = network.output_size();
  std::vector<double> output(static_cast<size_t>(out_size) * batch);
  network.copy_output_to_host(output.data(), output.size());

  long correct = 0;
  long total = batch;
  const double *target_ptr = targets.data();

  for (int i = 0; i < batch; ++i) {
    int pred_idx = 0;
    int true_idx = 0;
    double pred_max = output[i * out_size];
    double true_max = target_ptr[i * out_size];

    for (int r = 1; r < out_size; ++r) {
      double val = output[r + i * out_size];
      if (val > pred_max) {
        pred_max = val;
        pred_idx = r;
      }
      double tval = target_ptr[r + i * out_size];
      if (tval > true_max) {
        true_max = tval;
        true_idx = r;
      }
    }

    if (pred_idx == true_idx) {
      correct++;
    }
  }

  double accuracy = (static_cast<double>(correct) / total) * 100.0;
  double mse = 0.0;
  for (size_t i = 0; i < output.size(); ++i) {
    double diff = output[i] - target_ptr[i];
    mse += diff * diff;
  }
  mse *= 0.5;

  std::cout << "=== Test Results ===" << std::endl;
  std::cout << "Samples: " << total << std::endl;
  std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
  std::cout << "Total MSE: " << mse << std::endl;
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Eval time: " << elapsed.count() << " s" << std::endl;
  std::cout << "====================" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
  std::string config_path = "config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }
  SimpleConfig config = SimpleConfig::load(config_path);
  if (config.loaded()) {
    std::cout << "Loaded config: " << config_path << std::endl;
  } else {
    std::cout << "Config not found, using defaults: " << config_path << std::endl;
  }

  int train_size = config.getInt("gpu_train_size", 5000);
  int test_size = config.getInt("gpu_test_size", 1000);
  int max_iters = config.getInt("gpu_max_iters", 500);
  double tolerance = config.getDouble("gpu_tolerance", 1e-4);

  std::cout << "Loading Training Data..." << std::endl;
  Mat train_x = MNISTLoader::loadImages("../tests/mnist/train-images.idx3-ubyte", train_size);
  Mat train_y = MNISTLoader::loadLabels("../tests/mnist/train-labels.idx1-ubyte", train_size);

  std::cout << "Loading Test Data..." << std::endl;
  Mat test_x = MNISTLoader::loadImages("../tests/mnist/t10k-images.idx3-ubyte", test_size);
  Mat test_y = MNISTLoader::loadLabels("../tests/mnist/t10k-labels.idx1-ubyte", test_size);

  cuda_mlp::CublasHandle handle;
  cuda_mlp::CudaNetwork network(handle);
  network.addLayer(784, 128, cuda_mlp::ActivationType::Tanh);
  network.addLayer(128, 10, cuda_mlp::ActivationType::Linear);
  network.bindParams();

  cuda_mlp::DeviceBuffer<double> d_train_x;
  cuda_mlp::DeviceBuffer<double> d_train_y;
  cuda_mlp::DeviceBuffer<double> d_test_x;
  cuda_mlp::DeviceBuffer<double> d_test_y;
  d_train_x.copy_from_host(train_x.data(), train_x.size());
  d_train_y.copy_from_host(train_y.data(), train_y.size());
  d_test_x.copy_from_host(test_x.data(), test_x.size());
  d_test_y.copy_from_host(test_y.data(), test_y.size());

  cuda_mlp::CudaLBFGS solver(handle);
  solver.setMaxIterations(max_iters);
  solver.setTolerance(tolerance);

  std::cout << "\nStarting Training..." << std::endl;
  auto train_start = std::chrono::steady_clock::now();
  solver.solve(network, d_train_x.data(), d_train_y.data(), train_size);
  auto train_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> train_elapsed = train_end - train_start;
  std::cout << "\nTraining time: " << train_elapsed.count() << " s" << std::endl;

  std::cout << "\nTRAINING SET RESULTS:" << std::endl;
  report_results(network, d_train_x, train_y, train_size);

  std::cout << "\nTEST SET RESULTS:" << std::endl;
  report_results(network, d_test_x, test_y, test_size);

  return 0;
}
