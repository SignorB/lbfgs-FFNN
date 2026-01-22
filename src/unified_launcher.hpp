#pragma once

/**
 * @file unified_launcher.hpp
 * @brief Training launcher that wires datasets, networks, and optimizers.
 */

#include "network_wrapper.hpp"
#include "unified_optimization.hpp" // Optimizer strategies and config/dataset types
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>

template <typename Backend> class UnifiedLauncher;

/**
 * @brief CPU launcher specialization.
 */
template <> class UnifiedLauncher<CpuBackend> {
public:
  UnifiedLauncher() = default;

  /**
   * @brief Add a layer to the CPU network.
   * @tparam In Input dimension.
   * @tparam Out Output dimension.
   * @tparam Activation Activation function type.
   */
  template <int In, int Out, typename Activation> void addLayer() { net_wrapper_.addLayer<In, Out, Activation>(); }

  /**
   * @brief Finalize parameters and internal buffers.
   * @details Allocates memory for weights and gradients based on added layers.
   */
  void buildNetwork() { net_wrapper_.bindParams(); }

  /**
   * @brief Attach the training/test dataset.
   * @param data The UnifiedDataset containing train/test splits.
   */
  void setData(const UnifiedDataset &data) { dataset_ = data; }

  /**
   * @brief Run training for the selected optimizer.
   * @param optimizer The optimization strategy to use.
   * @param config configuration parameters for the experiment.
   */
  void train(UnifiedOptimizer<CpuBackend> &optimizer, const UnifiedConfig &config) {
    std::cout << ">>> Running CPU Experiment: " << config.name << std::endl;
    if (config.reset_params) {
      net_wrapper_.bindParams(config.seed);
    }
    // Train on the configured dataset.
    optimizer.optimize(net_wrapper_, dataset_, config);
    // Evaluate on training data.
    net_wrapper_.getInternal().test(dataset_.train_x, dataset_.train_y, "Training Results");
  }

  /**
   * @brief Evaluate on test data.
   * @details Prints MSE and Accuracy metrics to stdout.
   */
  void test() { net_wrapper_.getInternal().test(dataset_.test_x, dataset_.test_y, "Test Results"); }

  /**
   * @brief Access the underlying wrapper.
   * @return Reference to the NetworkWrapper.
   */
  NetworkWrapper<CpuBackend> &getWrapper() { return net_wrapper_; }

private:
  NetworkWrapper<CpuBackend> net_wrapper_;
  UnifiedDataset dataset_;
};

#ifdef __CUDACC__
  #include "cuda/cublas_handle.cuh"

/**
 * @brief CUDA launcher specialization.
 */
template <> class UnifiedLauncher<CudaBackend> {
public:
  UnifiedLauncher() : net_wrapper_(handle_) {}

  /**
   * @brief Add a layer to the CUDA network.
   * @tparam In Input dimension.
   * @tparam Out Output dimension.
   * @tparam Activation Activation function type.
   */
  template <int In, int Out, typename Activation> void addLayer() { net_wrapper_.addLayer<In, Out, Activation>(); }

  /**
   * @brief Finalize parameters and internal buffers.
   * @details Allocates memory for weights and gradients on the GPU.
   */
  void buildNetwork() { net_wrapper_.bindParams(); }

  /**
   * @brief Upload dataset to device buffers.
   * @param data The UnifiedDataset containing train/test splits.
   */
  void setData(const UnifiedDataset &data) {
    dataset_ = data;

    // Host-to-device upload.
    auto upload = [](const Eigen::MatrixXd &mat, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dev_buf) {
      if constexpr (std::is_same<double, cuda_mlp::CudaScalar>::value) {
        dev_buf.copy_from_host((const cuda_mlp::CudaScalar *)mat.data(), mat.size());
      } else {

        std::vector<cuda_mlp::CudaScalar> temp(mat.size());
        const double *ptr = mat.data();
        for (size_t i = 0; i < static_cast<size_t>(mat.size()); ++i)
          temp[i] = static_cast<cuda_mlp::CudaScalar>(ptr[i]);
        dev_buf.copy_from_host(temp.data(), temp.size());
      }
    };

    upload(dataset_.train_x, d_train_x_);
    upload(dataset_.train_y, d_train_y_);
    upload(dataset_.test_x, d_test_x_);
    upload(dataset_.test_y, d_test_y_);

    std::cout << "Data Uploaded to GPU. Train: " << dataset_.train_x.cols() << " samples." << std::endl;
  }

  /**
   * @brief Run training for the selected optimizer.
   * @param optimizer The optimization strategy to use.
   * @param config Configuration parameters for the experiment.
   */
  void train(UnifiedOptimizer<CudaBackend> &optimizer, const UnifiedConfig &config) {
    std::cout << ">>> Running CUDA Experiment: " << config.name << std::endl;
    if (config.reset_params) {
      net_wrapper_.bindParams(config.seed);
    }
    // Train on device buffers.
    optimizer.optimize(handle_, net_wrapper_, dataset_, d_train_x_, d_train_y_, config);
    // Evaluate on training data.
    evaluate(dataset_.train_x, dataset_.train_y, d_train_x_, "Training Results");
  }

  /**
   * @brief Evaluate on test data.
   * @details Runs forward pass on GPU and computes metrics on host.
   */
  void test() { evaluate(dataset_.test_x, dataset_.test_y, d_test_x_, "Test Results"); }

private:
  /// @brief Compute accuracy and MSE from device outputs.
  void evaluate(const Eigen::MatrixXd &x,
      const Eigen::MatrixXd &y,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &d_x,
      const char *label) {
    int batch_size = static_cast<int>(x.cols());
    int out_dim = static_cast<int>(y.rows());

    auto &net = net_wrapper_.getInternal();
    net.forward_only(d_x.data(), batch_size);

    std::vector<cuda_mlp::CudaScalar> host_output(batch_size * out_dim);
    net.copy_output_to_host(host_output.data(), host_output.size());

    double mse = 0;
    long correct = 0;
    const double *target_ptr = y.data();

    for (int i = 0; i < batch_size; ++i) {
      int pred_idx = 0;
      int true_idx = 0;
      double pred_max = -1e20;
      double true_max = -1e20;

      for (int r = 0; r < out_dim; ++r) {
        int idx = r + i * out_dim;
        double val = host_output[idx];
        double tval = target_ptr[idx];

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
      if (pred_idx == true_idx) correct++;
    }

    mse /= (double)(batch_size * out_dim);
    double acc = ((double)correct / batch_size) * 100.0;
    std::cout << label << ": MSE=" << mse << ", Accuracy=" << acc << "%" << std::endl;
  }

  cuda_mlp::CublasHandle handle_;
  NetworkWrapper<CudaBackend> net_wrapper_;
  UnifiedDataset dataset_;
  cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> d_train_x_, d_train_y_, d_test_x_, d_test_y_;
};
#endif
