#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace cuda_mlp {

class CudaSGD : public CudaMinimizerBase {
public:
  explicit CudaSGD(CublasHandle &handle) : CudaMinimizerBase(handle) {}

  void setLearningRate(CudaScalar lr) { lr_ = lr; }
  void setMomentum(CudaScalar momentum) { momentum_ = momentum; }
  void setBatchSize(int batch_size) { batch_size_ = batch_size; }
  void setLearningRateDecay(CudaScalar rate, int step_size) {
    decay_rate_ = rate;
    decay_step_ = step_size;
  }

  void setDimensions(int input_dim, int output_dim) {
    input_dim_ = input_dim;
    output_dim_ = output_dim;
  }

  void solve(int n,
      CudaScalar *params,
      const CudaScalar *input,
      const CudaScalar *target,
      int total_samples,
      const LossGradFun &loss_grad) override {

    if (n <= 0 || params == nullptr) return;
    if (input_dim_ == 0 || output_dim_ == 0) {
      std::cerr << "Error: Dimensions not set for SGD. Call setDimensions() first.\n";
      return;
    }

    DeviceBuffer<CudaScalar> grad(n);
    DeviceBuffer<CudaScalar> velocity(n);

    if (momentum_ > 0.0f) {
      device_set_zero(velocity.data(), n);
    }

    CudaScalar current_lr = lr_;

    int num_batches = (total_samples + batch_size_ - 1) / batch_size_;

    for (int iter = 0; iter < max_iters_; ++iter) {
      if (decay_step_ > 0 && iter > 0 && iter % decay_step_ == 0) {
        current_lr *= decay_rate_;
      }

      CudaScalar epoch_loss = 0.0f;

      for (int b = 0; b < num_batches; ++b) {
        int start_idx = b * batch_size_;
        int current_batch_size = std::min(batch_size_, total_samples - start_idx);
        const CudaScalar *batch_input = input + (start_idx * input_dim_);
        const CudaScalar *batch_target = target + (start_idx * output_dim_);

        CudaScalar batch_loss_sum = loss_grad(params, grad.data(), batch_input, batch_target, current_batch_size);
        CudaScalar effective_lr = current_lr / static_cast<CudaScalar>(current_batch_size);

        if (momentum_ > 0.0f) {
          device_scal(handle_, n, momentum_, velocity.data());
          device_axpy(handle_, n, -effective_lr, grad.data(), velocity.data());
          device_axpy(handle_, n, 1.0f, velocity.data(), params);
        } else {
          device_axpy(handle_, n, -effective_lr, grad.data(), params);
        }

        epoch_loss += batch_loss_sum;
      }
    }
  }

private:
  CudaScalar lr_ = 0.01f, momentum_ = 0.9f;
  CudaScalar decay_rate_ = 1.0f;
  int decay_step_ = 0;

  int batch_size_ = 64;
  int input_dim_ = 0;
  int output_dim_ = 0;
};

} // namespace cuda_mlp