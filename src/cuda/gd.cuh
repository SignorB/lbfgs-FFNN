#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <iostream>

namespace cuda_mlp {

class CudaGD : public CudaMinimizerBase {
public:
  explicit CudaGD(CublasHandle &handle) : CudaMinimizerBase(handle) {}

  void setLearningRate(CudaScalar lr) { lr_ = lr; }

  void setMomentum(CudaScalar momentum) { momentum_ = momentum; }

  void solve(int n,
      CudaScalar *params,
      const CudaScalar *input,
      const CudaScalar *target,
      int total_samples,
      const LossGradFun &loss_grad) override {

    if (n <= 0 || params == nullptr) {
      last_iterations_ = 0;
      return;
    }

    last_iterations_ = 0;
    DeviceBuffer<CudaScalar> grad(n);

    DeviceBuffer<CudaScalar> velocity(n);
    if (momentum_ > 0.0f) {
      device_set_zero(velocity.data(), n);
    }

    if (recorder_) recorder_->reset();
    CudaScalar loss = loss_grad(params, grad.data(), input, target, total_samples);
    CudaScalar effective_lr = lr_;
    effective_lr /= static_cast<CudaScalar>(total_samples);
    loss /= static_cast<CudaScalar>(total_samples);

    int iterations_done = 0;
    for (int iter = 0; iter < max_iters_; ++iter) {
      CudaScalar grad_norm = device_nrm2(handle_, grad.data(), n);
      if (grad_norm < tol_) break;

      if (momentum_ > 0.0f) {
        device_scal(handle_, n, momentum_, velocity.data());
        device_axpy(handle_, n, -effective_lr, grad.data(), velocity.data());
        device_axpy(handle_, n, 1.0f, velocity.data(), params);
      } else {
        device_axpy(handle_, n, -effective_lr, grad.data(), params);
      }
      loss = loss_grad(params, grad.data(), input, target, total_samples);
      CudaScalar grad_norm_after = device_nrm2(handle_, grad.data(), n);
      if (recorder_) recorder_->record(iterations_done, loss, grad_norm_after);
      iterations_done++;
    }
    last_iterations_ = iterations_done;
  }

private:
  CudaScalar lr_ = 0.01f, momentum_ = 0.9f;
};

} // namespace cuda_mlp
