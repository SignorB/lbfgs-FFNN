#pragma once

#include "common.cuh"

namespace cuda_mlp {

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

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

  ~DeviceBuffer() { release(); }

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

  void release() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  size_t size() const { return size_; }

  void copy_from_host(const T *host, size_t n) {
    resize(n);
    cuda_check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
  }
  void copy_to_host(T *host, size_t n) const {
    assert(n <= size_);
    cuda_check(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
  }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
};

} // namespace cuda_mlp
