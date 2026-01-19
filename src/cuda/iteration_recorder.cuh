#pragma once

#include "common.cuh"
#include "device_buffer.cuh"
#include <algorithm>
#include <vector>

namespace cuda_mlp {

/// @brief Stores loss and gradient norm history on device
struct IterationRecorder {
public:
  /// @brief Allocate buffers for up to @p capacity iterations
  void init(int capacity) {
    if (capacity <= 0) return;
    capacity_ = capacity;
    loss_.resize(static_cast<size_t>(capacity));
    grad_norm_.resize(static_cast<size_t>(capacity));
    time_ms_.resize(static_cast<size_t>(capacity));
    size_ = 0;
  }

  /// @brief Clear the current recorded size
  void reset() { size_ = 0; }

  /**
   * @brief Record a loss and gradient norm at an index
   * @param idx Iteration index
   * @param loss Loss value
   * @param grad_norm Gradient norm value
   */
  void record(int idx, CudaScalar loss, CudaScalar grad_norm, CudaScalar time_ms = 0) {
    if (idx < 0 || idx >= capacity_) return;
    cuda_check(cudaMemcpy(loss_.data() + idx, &loss, sizeof(CudaScalar), cudaMemcpyHostToDevice), "record loss");
    cuda_check(
        cudaMemcpy(grad_norm_.data() + idx, &grad_norm, sizeof(CudaScalar), cudaMemcpyHostToDevice), "record grad_norm");
    cuda_check(
        cudaMemcpy(time_ms_.data() + idx, &time_ms, sizeof(CudaScalar), cudaMemcpyHostToDevice), "record time_ms");
    size_ = std::max(size_, idx + 1);
  }

  /// @brief Copy recorded values back to host vectors
  void copy_to_host(std::vector<CudaScalar> &loss_out, std::vector<CudaScalar> &grad_norm_out) const {
    loss_out.resize(size_);
    grad_norm_out.resize(size_);
    if (size_ == 0) return;
    loss_.copy_to_host(loss_out.data(), size_);
    grad_norm_.copy_to_host(grad_norm_out.data(), size_);
  }

  /// @brief Copy recorded values (including time) back to host vectors
  void copy_to_host(
      std::vector<CudaScalar> &loss_out, std::vector<CudaScalar> &grad_norm_out, std::vector<CudaScalar> &time_ms_out) const {
    loss_out.resize(size_);
    grad_norm_out.resize(size_);
    time_ms_out.resize(size_);
    if (size_ == 0) return;
    loss_.copy_to_host(loss_out.data(), size_);
    grad_norm_.copy_to_host(grad_norm_out.data(), size_);
    time_ms_.copy_to_host(time_ms_out.data(), size_);
  }

  /// @brief Current number of recorded entries
  int size() const { return size_; }

private:
  DeviceBuffer<CudaScalar> loss_;      ///< Device buffer of losses
  DeviceBuffer<CudaScalar> grad_norm_; ///< Device buffer of gradient norms
  DeviceBuffer<CudaScalar> time_ms_;   ///< Device buffer of cumulative time (ms)
  int capacity_ = 0;                   ///< Allocated capacity
  int size_ = 0;                       ///< Current number of recorded entries
};

} // namespace cuda_mlp
