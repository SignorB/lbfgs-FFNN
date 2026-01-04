#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

namespace cuda_mlp {

/**
 * @brief Check a cuBLAS call and abort on failure.
 * @param status cuBLAS status code.
 * @param msg Context message describing the call.
 */
inline void cublas_check(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error: " << msg << " -> " << status << std::endl;
    std::abort();
  }
}

/**
 * @brief RAII wrapper for a cuBLAS handle.
 */
class CublasHandle {
public:
  /** @brief Create a new cuBLAS handle. */
  CublasHandle() { cublas_check(cublasCreate(&handle_), "cublasCreate"); }
  /** @brief Destroy the cuBLAS handle. */
  ~CublasHandle() {
    if (handle_) {
      cublasDestroy(handle_);
    }
  }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  /** @brief Access the underlying cuBLAS handle. */
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

} // namespace cuda_mlp
