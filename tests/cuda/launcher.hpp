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

  int log_interval = 10;
};

template <typename Scalar> struct Dataset {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> train_x, train_y, test_x, test_y;
  int train_size() const { return static_cast<int>(train_x.cols()); }
  int test_size() const { return static_cast<int>(test_x.cols()); }
};
template <typename Scalar> struct EvalStats {
  Scalar mse;
  Scalar accuracy;
};

template <typename Scalar> class ExperimentRunner {
public:
  ExperimentRunner() : network_(handle_) {}

  void setData(const Dataset<Scalar> &data) {
    dataset_ = data;
    d_train_x_.copy_from_host(dataset_.train_x.data(), dataset_.train_x.size());
    d_train_y_.copy_from_host(dataset_.train_y.data(), dataset_.train_y.size());
    d_test_x_.copy_from_host(dataset_.test_x.data(), dataset_.test_x.size());
    d_test_y_.copy_from_host(dataset_.test_y.data(), dataset_.test_y.size());
    std::cout << "Data Loaded. Train: " << dataset_.train_size() << ", Test: " << dataset_.test_size() << std::endl;
  }

  void buildNetwork(const std::vector<LayerConfig> &layers) {
    for (const auto &layer : layers) {
      network_.addLayer(layer.input_dim, layer.output_dim, layer.activation);
    }
    network_.bindParams();
    std::cout << "Network Built. Params: " << network_.params_size() << std::endl;
  }

  void runExperiments(const std::vector<RunConfig> &configs, const std::string &summary_csv = "summary_results.csv") {
    std::ofstream summary_file;
    summary_file.open(summary_csv);
    summary_file << "RunName,Optimizer,TotalTime(s),FinalTrainLoss,FinalTrainAcc,FinalTestLoss,FinalTestAcc,Iterations\n";

    DeviceBuffer<Scalar> initial_params(network_.params_size());
    cudaMemcpy(
        initial_params.data(), network_.params_data(), network_.params_size() * sizeof(Scalar), cudaMemcpyDeviceToDevice);

    for (const auto &config : configs) {
      std::cout << "\n>>> Running: " << config.name << " [" << (config.optimizer == OptimizerType::LBFGS ? "LBFGS" : "GD")
                << "] <<<" << std::endl;

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

      std::string log_filename = config.name + "_history.csv";
      std::ofstream log_file(log_filename);
      log_file << "Iteration,Time(s),TrainLoss,TrainAcc,TestLoss,TestAcc\n";

      auto start_time = std::chrono::steady_clock::now();
      int iter_count = 0;

      auto loss_grad =
          [&](const Scalar *params, Scalar *grad, const Scalar *input, const Scalar *target, int batch) -> Scalar {
        Scalar loss = network_.compute_loss_and_grad(input, target, batch);
        device_copy(grad, network_.grads_data(), network_.params_size());

        if (iter_count % config.log_interval == 0 || iter_count == 0) {
          cudaDeviceSynchronize();
          auto now = std::chrono::steady_clock::now();
          double elapsed = std::chrono::duration<double>(now - start_time).count();
          Scalar test_loss = network_.compute_loss_and_grad(d_test_x_.data(), d_test_y_.data(), dataset_.test_size());
          Scalar grad_norm = 0;

          log_file << iter_count << "," << elapsed << "," << loss << "," << test_loss << "," << grad_norm << "\n";
          if (iter_count % 100 == 0) {
            std::cout << "Iter " << iter_count << " | T: " << elapsed << "s | Loss: " << loss << std::endl;
          }
        }
        iter_count++;
        return loss;
      };
      solver->solve(network_.params_size(),
          network_.params_data(),
          d_train_x_.data(),
          d_train_y_.data(),
          dataset_.train_size(),
          loss_grad);
      cudaDeviceSynchronize();
      auto end_time = std::chrono::steady_clock::now();
      double total_time = std::chrono::duration<double>(end_time - start_time).count();

      EvalStats<Scalar> final_train = evaluate(d_train_x_, dataset_.train_y);
      EvalStats<Scalar> final_test = evaluate(d_test_x_, dataset_.test_y);

      summary_file << config.name << "," << (config.optimizer == OptimizerType::LBFGS ? "LBFGS" : "GD") << "," << total_time
                   << "," << final_train.mse << "," << final_train.accuracy << "," << final_test.mse << ","
                   << final_test.accuracy << "," << iter_count << "\n";

      log_file.close();
      std::cout << "Done: " << config.name << " (Test Acc: " << final_test.accuracy << "%)\n";
    }
    summary_file.close();
  }

private:
  EvalStats<Scalar> evaluate(
      const DeviceBuffer<Scalar> &d_input, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &host_targets) {
    int batch_size = static_cast<int>(host_targets.cols());
    int out_dim = static_cast<int>(host_targets.rows());

    network_.forward_only(d_input.data(), batch_size);
    std::vector<Scalar> host_output(batch_size * out_dim);
    network_.copy_output_to_host(host_output.data(), host_output.size());
    Scalar mse = 0;
    long correct = 0;
    const Scalar *target_ptr = host_targets.data();

    for (int i = 0; i < batch_size; ++i) {
      int pred_idx = 0;
      int true_idx = 0;
      Scalar pred_max = -1e20;
      Scalar true_max = -1e20;
      for (int r = 0; r < out_dim; ++r) {
        int idx = r + i * out_dim;
        Scalar val = host_output[idx];
        Scalar tval = target_ptr[idx];
        mse += (val - tval) * (val - tval);
        if (val > pred_max) {
          pred_max = val;
          pred_idx = r;
        }
        if (tval > true_max) {
          true_max = tval;
          true_idx = r;
        }
      }
      if (pred_idx == true_idx) {
        correct++;
      }
    }
    EvalStats<Scalar> stats;
    stats.mse = mse / (Scalar)(batch_size * out_dim);
    stats.accuracy = ((Scalar)correct / (Scalar)batch_size) * 100.0f;
    return stats;
  }

  CublasHandle handle_;
  CudaNetwork network_;
  Dataset<Scalar> dataset_;
  DeviceBuffer<Scalar> d_train_x_, d_train_y_, d_test_x_, d_test_y_;
};

} // namespace cuda_mlp