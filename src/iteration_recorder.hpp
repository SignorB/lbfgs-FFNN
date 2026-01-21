#pragma once

#include <algorithm>
#include <vector>

struct CpuBackend;
struct CudaBackend;

/**
 * @brief Iteration recorder specialized by backend (CPU/CUDA).
 * @tparam Backend Backend tag type.
 */
template <typename Backend> class IterationRecorder;

/**
 * @brief CPU recorder that stores loss/gradient history on host.
 */
template <> class IterationRecorder<CpuBackend> {
public:
  /// @brief Allocate buffers for up to @p capacity iterations.
  void init(int capacity) {
    if (capacity <= 0) return;
    capacity_ = capacity;
    loss_.assign(static_cast<size_t>(capacity), 0.0);
    grad_norm_.assign(static_cast<size_t>(capacity), 0.0);
    time_ms_.assign(static_cast<size_t>(capacity), 0.0);
    size_ = 0;
  }

  /// @brief Reset recorded size without releasing memory.
  void reset() { size_ = 0; }

  /**
   * @brief Record a loss/grad/time entry at iteration index.
   * @param idx Iteration index.
   * @param loss Loss value.
   * @param grad_norm Gradient norm value.
   * @param time_ms Cumulative time in ms.
   */
  void record(int idx, double loss, double grad_norm, double time_ms = 0.0) {
    if (idx < 0 || idx >= capacity_) return;
    size_t i = static_cast<size_t>(idx);
    loss_[i] = loss;
    grad_norm_[i] = grad_norm;
    time_ms_[i] = time_ms;
    size_ = std::max(size_, idx + 1);
  }

  /// @brief Copy recorded loss and gradient norm to output vectors.
  void copy_to_host(std::vector<double> &loss_out, std::vector<double> &grad_norm_out) const {
    loss_out.assign(loss_.begin(), loss_.begin() + size_);
    grad_norm_out.assign(grad_norm_.begin(), grad_norm_.begin() + size_);
  }

  /// @brief Copy recorded loss, gradient norm, and time to output vectors.
  void copy_to_host(
      std::vector<double> &loss_out, std::vector<double> &grad_norm_out, std::vector<double> &time_ms_out) const {
    loss_out.assign(loss_.begin(), loss_.begin() + size_);
    grad_norm_out.assign(grad_norm_.begin(), grad_norm_.begin() + size_);
    time_ms_out.assign(time_ms_.begin(), time_ms_.begin() + size_);
  }

  /// @brief Current number of recorded entries.
  int size() const { return size_; }

private:
  std::vector<double> loss_;      ///< Loss values per iteration.
  std::vector<double> grad_norm_; ///< Gradient norms per iteration.
  std::vector<double> time_ms_;   ///< Cumulative time in ms.
  int capacity_ = 0;              ///< Allocated capacity.
  int size_ = 0;                  ///< Current number of entries.
};

#ifdef __CUDACC__
  #include "cuda/common.cuh"
  #include "cuda/device_buffer.cuh"

/**
 * @brief CUDA recorder that stores history on device and copies to host on demand.
 */
template <> class IterationRecorder<CudaBackend> {
public:
  /// @brief Allocate device buffers for up to @p capacity iterations.
  void init(int capacity) {
    if (capacity <= 0) return;
    capacity_ = capacity;
    loss_.resize(static_cast<size_t>(capacity));
    grad_norm_.resize(static_cast<size_t>(capacity));
    time_ms_.resize(static_cast<size_t>(capacity));
    size_ = 0;
  }

  /// @brief Reset recorded size without releasing device memory.
  void reset() { size_ = 0; }

  /**
   * @brief Record a loss/grad/time entry at iteration index (device write).
   * @param idx Iteration index.
   * @param loss Loss value.
   * @param grad_norm Gradient norm value.
   * @param time_ms Cumulative time in ms.
   */
  void record(int idx, cuda_mlp::CudaScalar loss, cuda_mlp::CudaScalar grad_norm, cuda_mlp::CudaScalar time_ms = 0) {
    if (idx < 0 || idx >= capacity_) return;
    cuda_mlp::cuda_check(
        cudaMemcpy(loss_.data() + idx, &loss, sizeof(cuda_mlp::CudaScalar), cudaMemcpyHostToDevice), "record loss");
    cuda_mlp::cuda_check(
        cudaMemcpy(grad_norm_.data() + idx, &grad_norm, sizeof(cuda_mlp::CudaScalar), cudaMemcpyHostToDevice),
        "record grad_norm");
    cuda_mlp::cuda_check(
        cudaMemcpy(time_ms_.data() + idx, &time_ms, sizeof(cuda_mlp::CudaScalar), cudaMemcpyHostToDevice), "record time_ms");
    size_ = std::max(size_, idx + 1);
  }

  /// @brief Copy recorded loss and gradient norm to host vectors.
  void copy_to_host(std::vector<cuda_mlp::CudaScalar> &loss_out, std::vector<cuda_mlp::CudaScalar> &grad_norm_out) const {
    loss_out.resize(size_);
    grad_norm_out.resize(size_);
    if (size_ == 0) return;
    loss_.copy_to_host(loss_out.data(), size_);
    grad_norm_.copy_to_host(grad_norm_out.data(), size_);
  }

  /// @brief Copy recorded loss, gradient norm, and time to host vectors.
  void copy_to_host(std::vector<cuda_mlp::CudaScalar> &loss_out,
      std::vector<cuda_mlp::CudaScalar> &grad_norm_out,
      std::vector<cuda_mlp::CudaScalar> &time_ms_out) const {
    loss_out.resize(size_);
    grad_norm_out.resize(size_);
    time_ms_out.resize(size_);
    if (size_ == 0) return;
    loss_.copy_to_host(loss_out.data(), size_);
    grad_norm_.copy_to_host(grad_norm_out.data(), size_);
    time_ms_.copy_to_host(time_ms_out.data(), size_);
  }

  /// @brief Current number of recorded entries.
  int size() const { return size_; }

private:
  cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> loss_;      ///< Device loss buffer.
  cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> grad_norm_; ///< Device gradient norm buffer.
  cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> time_ms_;   ///< Device time buffer.
  int capacity_ = 0;                                       ///< Allocated capacity.
  int size_ = 0;                                           ///< Current number of entries.
};
#endif
