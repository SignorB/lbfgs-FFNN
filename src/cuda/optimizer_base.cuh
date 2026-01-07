#pragma once

#include "cublas_handle.cuh"
#include "network.cuh"

namespace cuda_mlp {

class CudaOptimizer {
public:
  explicit CudaOptimizer(CublasHandle &handle) : handle_(handle) {}
  virtual ~CudaOptimizer() = default;

  void setMaxIterations(int iters) { max_iters_ = iters; }
  void setTolerance(CudaScalar tol) { tol_ = tol; }

  virtual void solve(CudaNetwork &net, const CudaScalar *input, const CudaScalar *target, int batch) = 0;

protected:
  CublasHandle &handle_;
  int max_iters_ = 200;
  CudaScalar tol_ = 1e-6f;
};

} // namespace cuda_mlp
