#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace cuda_mlp {

/// @brief SGD with optional momentum and learning-rate decay
class CudaSGD : public CudaMinimizerBase {
public:
  /// @brief Construct the optimizer
  explicit CudaSGD(CublasHandle &handle) : CudaMinimizerBase(handle) {}

  /// @brief Set the base learning rate
  void setLearningRate(CudaScalar lr) { lr_ = lr; }
  /// @brief Set the momentum factor in [0,1)
  void setMomentum(CudaScalar momentum) { momentum_ = momentum; }
  /// @brief Set minibatch size
  void setBatchSize(int batch_size) { batch_size_ = batch_size; }
  /**
   * @brief Configure step-wise learning rate decay
   * @param rate Multiplicative decay factor
   * @param step_size Iterations between decays
   */
  void setLearningRateDecay(CudaScalar rate, int step_size) {
    decay_rate_ = rate;
    decay_step_ = step_size;
  }

  /// @brief Set the input/output dimensions to stride batches
  void setDimensions(int input_dim, int output_dim) {
    input_dim_ = input_dim;
    output_dim_ = output_dim;
  }

  /**
   * @brief Run SGD optimization
   * @param n Number of parameters
   * @param params Parameter vector (device)
   * @param input Input data (device)
   * @param target Target data (device)
   * @param total_samples Total number of samples
   * @param loss_grad Callback returning batch loss and gradient
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
    if (input_dim_ == 0 || output_dim_ == 0) {
      std::cerr << "Error: Dimensions not set for SGD\n";
      last_iterations_ = 0;
      return;
    }

    last_iterations_ = 0;
    DeviceBuffer<CudaScalar> grad(n);
    DeviceBuffer<CudaScalar> velocity(n);

    if (momentum_ > 0.0f) device_set_zero(velocity.data(), n);

    if (recorder_) recorder_->reset();

    CudaScalar current_lr = lr_;
    int num_batches = (total_samples + batch_size_ - 1) / batch_size_;
    CudaScalar prev_epoch_loss_avg = std::numeric_limits<CudaScalar>::infinity();

    int iterations_done = 0;
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
          // v = momentum * v - lr * grad; params += v
          device_scal(handle_, n, momentum_, velocity.data());
          device_axpy(handle_, n, -effective_lr, grad.data(), velocity.data());
          device_axpy(handle_, n, 1.0f, velocity.data(), params);
        } else {
          // params -= lr * grad
          device_axpy(handle_, n, -effective_lr, grad.data(), params);
        }

        epoch_loss += batch_loss_sum;
      }

      CudaScalar epoch_loss_avg = epoch_loss / static_cast<CudaScalar>(total_samples);
      if (tol_ > static_cast<CudaScalar>(0) && std::isfinite(prev_epoch_loss_avg)) {
        CudaScalar denom = std::max(static_cast<CudaScalar>(1), std::abs(prev_epoch_loss_avg));
        CudaScalar rel_impr = std::abs(prev_epoch_loss_avg - epoch_loss_avg) / denom;

        if (rel_impr < tol_) break;
      }

      prev_epoch_loss_avg = epoch_loss_avg;
      CudaScalar grad_norm = device_nrm2(handle_, grad.data(), n);
      if (recorder_) recorder_->record(iterations_done, epoch_loss_avg, grad_norm);
      iterations_done++;
    }
    last_iterations_ = iterations_done;
  }

private:
  CudaScalar lr_ = 0.01f;        ///< Base learning rate
  CudaScalar momentum_ = 0.9f;   ///< Momentum factor
  CudaScalar decay_rate_ = 1.0f; ///< Multiplicative decay factor
  int decay_step_ = 0;           ///< Decay step interval

  int batch_size_ = 64; ///< Minibatch size
  int input_dim_ = 0;   ///< Input dimension (for batch slicing)
  int output_dim_ = 0;  ///< Output dimension (for batch slicing)
};

} // namespace cuda_mlp
