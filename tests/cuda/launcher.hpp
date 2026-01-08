#pragma once

#include "../../src/cuda/gd.cuh"
#include "../../src/cuda/lbfgs.cuh"
#include "../../src/cuda/network.cuh"
#include "../cuda_report.hpp"
#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace cuda_mlp {

enum class OptimizerType { GD, LBFGS };

struct LayerConfig {
  int input_dim;
  int output_dim;
  ActivationType activation;
};

struct RunConfig {
  std::string name;
  OptimizerType optimizer;
  int max_iters = 200;
  float tolerance = 1e-4f;

  size_t lbfgs_memory = 10;

  float gd_lr = 0.01f, gd_momentum = 0.9f;
};

template <typename Scalar> struct Dataset {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> train_x, train_y, test_x, test_y;

  int train_size() const { return static_cast<int>(train_x.cols()); }
  int test_size() const { return static_cast<int>(test_x.cols()); }
  int input_dim() const { return static_cast<int>(train_x.rows()); }
  int output_dim() const { return static_cast<int>(train_y.rows()); }
};

template <typename Scalar> class ExperimentRunner {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

public:
  ExperimentRunner() : network_(handle_) {}

  void setData(const Dataset<Scalar> &data) {
    dataset_ = data;
    d_train_x_.copy_from_host(dataset_.train_x.data(), dataset_.train_x.size());
    d_train_y_.copy_from_host(dataset_.train_y.data(), dataset_.train_y.size());
    d_test_x_.copy_from_host(dataset_.test_x.data(), dataset_.test_x.size());
    d_test_y_.copy_from_host(dataset_.test_y.data(), dataset_.test_y.size());

    std::cout << "Data Loaded on GPU. Train: " << dataset_.train_size() << ", Test: " << dataset_.test_size() << std::endl;
  }

  void buildNetwork(const std::vector<LayerConfig> &layers) {
    for (const auto &layer : layers) {
      network_.addLayer(layer.input_dim, layer.output_dim, layer.activation);
    }
    network_.bindParams();
    std::cout << "Network Built. Params size: " << network_.params_size() << std::endl;
  }

  void runExperiments(const std::vector<RunConfig> &configs, const std::string &output_csv = "results.csv") {
    std::ofstream csv_file;
    csv_file.open(output_csv);
    csv_file << "RunName,Optimizer,Time(s),FinalTrainLoss,FinalTestMSE,Iterations\n";

    DeviceBuffer<Scalar> initial_params(network_.params_size());

    cudaMemcpy(
        initial_params.data(), network_.params_data(), network_.params_size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);

    for (const auto &config : configs) {
      std::cout << "\n>>> Running: " << config.name << " <<<" << std::endl;

      cudaMemcpy(
          network_.params_data(), initial_params.data(), network_.params_size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);

      std::unique_ptr<CudaMinimizerBase> solver;

      if (config.optimizer == OptimizerType::LBFGS) {
        auto lbfgs = std::make_unique<CudaLBFGS>(handle_);
        lbfgs->setMemory(config.lbfgs_memory);
        lbfgs->setMaxIterations(config.max_iters);
        lbfgs->setTolerance(config.tolerance);
        solver = std::move(lbfgs);
      } else {
        auto gd = std::make_unique<CudaGD>(handle_);
        gd->setLearningRate(config.gd_lr);
        gd->setMomentum(config.gd_momentum);
        gd->setMaxIterations(config.max_iters);
        gd->setTolerance(config.tolerance);
        solver = std::move(gd);
      }

      auto start_time = std::chrono::steady_clock::now();

      const int params_size = static_cast<int>(network_.params_size());
      std::vector<Scalar> loss_history;
      loss_history.reserve(static_cast<size_t>(config.max_iters));

      auto loss_grad =
          [&](const Scalar *params, Scalar *grad, const Scalar *input, const Scalar *target, int batch) -> Scalar {
        (void)params;
        Scalar loss = network_.compute_loss_and_grad(input, target, batch);
        device_copy(grad, network_.grads_data(), params_size);

        loss_history.push_back(loss);
        return loss;
      };

      solver->solve(
          params_size, network_.params_data(), d_train_x_.data(), d_train_y_.data(), dataset_.train_size(), loss_grad);

      auto end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;

      std::cout << "Time: " << elapsed.count() << "s" << std::endl;

      Scalar train_loss = network_.compute_loss_and_grad(d_train_x_.data(), d_train_y_.data(), dataset_.train_size());
      Scalar test_loss = network_.compute_loss_and_grad(d_test_x_.data(), d_test_y_.data(), dataset_.test_size());

      csv_file << config.name << "," << (config.optimizer == OptimizerType::LBFGS ? "LBFGS" : "GD") << "," << elapsed.count()
               << "," << train_loss << "," << test_loss << "," << config.max_iters << "\n";

      std::ofstream loss_file(config.name + "_loss.csv");
      for (size_t i = 0; i < loss_history.size(); ++i) {
        loss_file << i << "," << loss_history[i] << "\n";
      }

    }

    csv_file.close();
    std::cout << "\nResults saved to " << output_csv << std::endl;
  }

private:
  CublasHandle handle_;
  CudaNetwork network_;
  Dataset<Scalar> dataset_;

  DeviceBuffer<Scalar> d_train_x_, d_train_y_, d_test_x_, d_test_y_;
};

} // namespace cuda_mlp
