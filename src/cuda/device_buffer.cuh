#pragma once
#include "common.cuh"

namespace cuda_mlp {

/// @brief Owning buffer for device memory
template <typename T> class DeviceBuffer {
public:
  /// @brief Construct an empty buffer.
  DeviceBuffer() = default;
  /**
   * @brief Construct and allocate a buffer of size @p n.
   * @param n Number of elements.
   */
  explicit DeviceBuffer(size_t n) { resize(n); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  /// @brief Move-construct, transferring ownership.
  DeviceBuffer(DeviceBuffer &&other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  /// @brief Move-assign, transferring ownership and releasing previous memory.
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

  /// @brief Release any owned device memory.
  ~DeviceBuffer() { release(); }

  /**
   * @brief Resize the buffer, reallocating if needed.
   * @param n New element count.
   */
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

  /// @brief Free device memory, if allocated.
  void release() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  /// @brief Mutable raw pointer to device memory.
  T *data() { return ptr_; }
  /// @brief Const raw pointer to device memory.
  const T *data() const { return ptr_; }
  /// @brief Current number of elements.
  size_t size() const { return size_; }

  /**
   * @brief Copy from host to device, resizing as needed.
   * @param host Host pointer.
   * @param n Number of elements.
   */
  void copy_from_host(const T *host, size_t n) {
    resize(n);
    cuda_check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
  }
  /**
   * @brief Copy from device to host.
   * @param host Host pointer to write into.
   * @param n Number of elements to copy.
   */
  void copy_to_host(T *host, size_t n) const {
    assert(n <= size_);
    cuda_check(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
  }

private:
  T *ptr_ = nullptr; ///< Device pointer.
  size_t size_ = 0;  ///< Element count.
};

} // namespace cuda_mlp
