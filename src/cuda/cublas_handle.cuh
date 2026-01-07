#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

namespace cuda_mlp {

inline void cublas_check(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error: " << msg << " -> " << status << std::endl;
    std::abort();
  }
}
class CublasHandle {
public:
  CublasHandle() { cublas_check(cublasCreate(&handle_), "cublasCreate"); }
  ~CublasHandle() {
    if (handle_) {
      cublasDestroy(handle_);
    }
  }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

} // namespace cuda_mlp
