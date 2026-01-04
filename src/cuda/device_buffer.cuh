#pragma once

#include "common.cuh"

namespace cuda_mlp {

/**
 * @brief Simple device buffer with ownership and copy utilities.
 * @tparam T Element type stored on device.
 */
template <typename T> class DeviceBuffer {
public:
  /** @brief Construct an empty buffer. */
  DeviceBuffer() = default;
  /** @brief Construct a buffer with @p n elements allocated on device. */
  explicit DeviceBuffer(size_t n) { resize(n); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  /** @brief Move construct from another buffer. */
  DeviceBuffer(DeviceBuffer &&other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  /** @brief Move-assign from another buffer. */
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      release();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /** @brief Destroy the buffer and release device memory. */
  ~DeviceBuffer() { release(); }

  /** @brief Resize the buffer to @p n elements, reallocating if needed. */
  void resize(size_t n) {
    if (n == size_) {
      return;
    }
    release();
    if (n > 0) {
      cuda_check(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc");
      size_ = n;
    }
  }

  /** @brief Release device memory, leaving the buffer empty. */
  void release() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  /** @brief Get the raw device pointer. */
  T *data() { return ptr_; }
  /** @brief Get the raw device pointer (const). */
  const T *data() const { return ptr_; }
  /** @brief Get the number of elements in the buffer. */
  size_t size() const { return size_; }

  /** @brief Copy @p n elements from host to device, resizing as needed. */
  void copy_from_host(const T *host, size_t n) {
    resize(n);
    cuda_check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
  }

  /** @brief Copy @p n elements from device to host. */
  void copy_to_host(T *host, size_t n) const {
    assert(n <= size_);
    cuda_check(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
  }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
};

} // namespace cuda_mlp
