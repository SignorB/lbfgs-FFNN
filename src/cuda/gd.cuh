#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <iostream>

namespace cuda_mlp {

/**
 * @brief Gradient descent with optional momentum
 * @details
 *   Minimizes a loss L(theta) using updates:
 *   - Without momentum: theta_{k+1} = theta_k - lr * grad(L).
 *   - With momentum: v_{k+1} = m * v_k - lr * grad(L), theta_{k+1} = theta_k + v_{k+1}.
 */
class CudaGD : public CudaMinimizerBase {
public:
  /// @brief Construct the optimizer
  explicit CudaGD(CublasHandle &handle) : CudaMinimizerBase(handle) {}

  /// @brief Set the learning rate
  void setLearningRate(CudaScalar lr) { lr_ = lr; }

  /// @brief Set the momentum factor in [0,1)
  void setMomentum(CudaScalar momentum) { momentum_ = momentum; }

  /**
   * @brief Run full-batch gradient descent
   * @param n Number of parameters
   * @param params Parameter vector (device)
   * @param input Input batch (device)
   * @param target Target batch (device)
   * @param total_samples Number of samples in the full batch
   * @param loss_grad Callback returning loss and gradient
   */
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
    // Full-batch gradient and loss. loss_grad returns sum over samples.
    CudaScalar loss = loss_grad(params, grad.data(), input, target, total_samples);
    // Scale by batch size: we want average gradient and average loss.
    CudaScalar effective_lr = lr_ / static_cast<CudaScalar>(total_samples);
    loss /= static_cast<CudaScalar>(total_samples);

    cudaEvent_t iter_start{};
    cudaEvent_t iter_stop{};
    CudaScalar elapsed_ms = 0.0f;
    const bool timing = (recorder_ != nullptr);
    if (timing) {
      cuda_check(cudaEventCreate(&iter_start), "cudaEventCreate iter_start");
      cuda_check(cudaEventCreate(&iter_stop), "cudaEventCreate iter_stop");
    }

    int iterations_done = 0;
    for (int iter = 0; iter < max_iters_; ++iter) {
      if (timing) cuda_check(cudaEventRecord(iter_start), "cudaEventRecord iter_start");
      CudaScalar grad_norm = device_nrm2(handle_, grad.data(), n);
      if (grad_norm < tol_) break;

      if (momentum_ > 0.0f) {
        // Momentum update:
        // v <- m * v - lr * g
        // x <- x + v
        device_scal(handle_, n, momentum_, velocity.data());
        device_axpy(handle_, n, -effective_lr, grad.data(), velocity.data());
        device_axpy(handle_, n, 1.0f, velocity.data(), params);
      } else {
        // Plain GD: x <- x - lr * g
        device_axpy(handle_, n, -effective_lr, grad.data(), params);
      }
      loss = loss_grad(params, grad.data(), input, target, total_samples);
      CudaScalar grad_norm_after = device_nrm2(handle_, grad.data(), n);
      if (timing) {
        cuda_check(cudaEventRecord(iter_stop), "cudaEventRecord iter_stop");
        cuda_check(cudaEventSynchronize(iter_stop), "cudaEventSynchronize iter_stop");
        float iter_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&iter_ms, iter_start, iter_stop), "cudaEventElapsedTime iter");
        elapsed_ms += static_cast<CudaScalar>(iter_ms);
      }
      if (recorder_) recorder_->record(iterations_done, loss, grad_norm_after, elapsed_ms);
      iterations_done++;
    }
    if (timing) {
      cuda_check(cudaEventDestroy(iter_start), "cudaEventDestroy iter_start");
      cuda_check(cudaEventDestroy(iter_stop), "cudaEventDestroy iter_stop");
    }
    last_iterations_ = iterations_done;
  }

private:
  CudaScalar lr_ = 0.01f;      ///< Base learning rate
  CudaScalar momentum_ = 0.9f; ///< Momentum factor
};

} // namespace cuda_mlp
