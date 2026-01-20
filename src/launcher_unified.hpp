#pragma once

#include "network_wrapper.hpp"
#include "unified_optimization.hpp" // Includes Optimizer strategies and Config/Dataset structs
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <vector>


template <typename Backend> class UnifiedLauncher;


template <> class UnifiedLauncher<CpuBackend> {
public:
  UnifiedLauncher() = default;

  
  template <int In, int Out, typename Activation> void addLayer() { net_wrapper_.addLayer<In, Out, Activation>(); }

  void buildNetwork() { net_wrapper_.bindParams(); }

  void setData(const UnifiedDataset &data) { dataset_ = data; }

  
  void train(UnifiedOptimizer<CpuBackend>& optimizer, const UnifiedConfig &config) {
    std::cout << ">>> Running CPU Experiment: " << config.name << std::endl;
    
    
    optimizer.optimize(net_wrapper_, dataset_, config);

    
    net_wrapper_.getInternal().test(dataset_.train_x, dataset_.train_y, "Training Results");
  }

  NetworkWrapper<CpuBackend> &getWrapper() { return net_wrapper_; }

private:
  NetworkWrapper<CpuBackend> net_wrapper_;
  UnifiedDataset dataset_;
};


#ifdef __CUDACC__
#include "cuda/cublas_handle.cuh"

template <> class UnifiedLauncher<CudaBackend> {
public:
  UnifiedLauncher() : net_wrapper_(handle_) {}

  template <int In, int Out, typename Activation> void addLayer() { net_wrapper_.addLayer<In, Out, Activation>(); }

  void buildNetwork() { net_wrapper_.bindParams(); }

  void setData(const UnifiedDataset &data) {
    dataset_ = data;
    
    
    auto upload = [](const Eigen::MatrixXd& mat, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dev_buf) {
        if constexpr (std::is_same<double, cuda_mlp::CudaScalar>::value) {
            dev_buf.copy_from_host((const cuda_mlp::CudaScalar*)mat.data(), mat.size());
        } else {
            
            std::vector<cuda_mlp::CudaScalar> temp(mat.size());
            const double* ptr = mat.data();
            for(size_t i=0; i<mat.size(); ++i) temp[i] = static_cast<cuda_mlp::CudaScalar>(ptr[i]);
            dev_buf.copy_from_host(temp.data(), temp.size());
        }
    };

    upload(dataset_.train_x, d_train_x_);
    upload(dataset_.train_y, d_train_y_);
    upload(dataset_.test_x, d_test_x_);
    upload(dataset_.test_y, d_test_y_);

    std::cout << "Data Uploaded to GPU. Train: " << dataset_.train_x.cols() << " samples." << std::endl;
  }

  
  void train(UnifiedOptimizer<CudaBackend>& optimizer, const UnifiedConfig &config) {
    std::cout << ">>> Running CUDA Experiment: " << config.name << std::endl;

    
    optimizer.optimize(handle_, net_wrapper_, dataset_, d_train_x_, d_train_y_, config);

    evaluate();
  }

  void evaluate() {
      int batch_size = static_cast<int>(dataset_.test_x.cols());
      int out_dim = static_cast<int>(dataset_.test_y.rows());

      auto &net = net_wrapper_.getInternal();
      net.forward_only(d_test_x_.data(), batch_size);

      std::vector<cuda_mlp::CudaScalar> host_output(batch_size * out_dim);
      net.copy_output_to_host(host_output.data(), host_output.size());

      double mse = 0;
      long correct = 0;
      const double *target_ptr = dataset_.test_y.data();

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

          if (val > pred_max) { pred_max = val; pred_idx = r; }
          if (tval > true_max) { true_max = tval; true_idx = r; }
        }
        if (pred_idx == true_idx) correct++;
      }

      mse /= (double)(batch_size * out_dim);
      double acc = ((double)correct / batch_size) * 100.0;
      std::cout << "Test Results: MSE=" << mse << ", Accuracy=" << acc << "%" << std::endl;
  }

private:
  cuda_mlp::CublasHandle handle_;
  NetworkWrapper<CudaBackend> net_wrapper_;
  UnifiedDataset dataset_;
  cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> d_train_x_, d_train_y_, d_test_x_, d_test_y_;
};
#endif
