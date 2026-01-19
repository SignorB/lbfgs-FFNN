#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

namespace cuda_mlp {

/**
 * @brief Check a cuBLAS API call and abort with a message on failure.
 * @param status cuBLAS status code.
 * @param msg Context string describing the operation.
 */
inline void cublas_check(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error: " << msg << " -> " << status << std::endl;
    std::abort();
  }
}

/// @brief RAII-managed cuBLAS handle.
class CublasHandle {
public:
  /// @brief Construct and initialize the cuBLAS handle.
  CublasHandle() { cublas_check(cublasCreate(&handle_), "cublasCreate"); }
  /// @brief Destroy the handle if it exists.
  ~CublasHandle() {
    if (handle_) {
      cublasDestroy(handle_);
    }
  }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  /// @brief Access the raw cuBLAS handle.
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr; ///< Owned cuBLAS handle.
};

} // namespace cuda_mlp
