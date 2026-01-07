#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "optimizer_base.cuh"
#include <cmath>
#include <iostream>

namespace cuda_mlp {

class CudaSGD : public CudaOptimizer {
public:
  explicit CudaSGD(CublasHandle &handle) : CudaOptimizer(handle) {}

  void setLearningRate(CudaScalar lr) { learning_rate_ = lr; }
  void setMomentum(CudaScalar momentum) { momentum_ = momentum; }
  void setWeightDecay(CudaScalar weight_decay) { weight_decay_ = weight_decay; }
  void setReportEvery(int iters) { report_every_ = iters; }

  void solve(CudaNetwork &net, const CudaScalar *input, const CudaScalar *target, int batch) override {
    const int n = static_cast<int>(net.params_size());
    if (n <= 0) {
      return;
    }

    CudaScalar *params = net.params_data();
    CudaScalar *grads = net.grads_data();

    DeviceBuffer<CudaScalar> velocity;
    if (momentum_ != 0.0f) {
      velocity.resize(static_cast<size_t>(n));
      device_set_zero(velocity.data(), static_cast<size_t>(n));
    }

    for (int iter = 0; iter < max_iters_; ++iter) {
      CudaScalar loss = net.compute_loss_and_grad(input, target, batch);

      if (weight_decay_ != 0.0f) {
        device_axpy(handle_, n, weight_decay_, params, grads);
      }

      CudaScalar grad_norm = device_nrm2(handle_, grads, n);
      if (!std::isfinite(loss) || !std::isfinite(grad_norm)) {
        std::cerr << "SGD aborted: non-finite loss/grad_norm at iter " << (iter + 1) << std::endl;
        break;
      }
      if (grad_norm < tol_) {
        break;
      }

      if (momentum_ == 0.0f) {
        device_axpy(handle_, n, -learning_rate_, grads, params);
      } else {
        device_scal(handle_, n, momentum_, velocity.data());
        device_axpy(handle_, n, 1.0f, grads, velocity.data());
        device_axpy(handle_, n, -learning_rate_, velocity.data(), params);
      }

      if (report_every_ > 0 && ((iter + 1) % report_every_ == 0)) {
        std::cout << "Iter " << (iter + 1) << " - Loss: " << loss << " - |g|: " << grad_norm << std::endl;
      }
    }
  }

private:
  CudaScalar learning_rate_ = 1e-2f;
  CudaScalar momentum_ = 0.0f;
  CudaScalar weight_decay_ = 0.0f;
  int report_every_ = 1;
};

} // namespace cuda_mlp
