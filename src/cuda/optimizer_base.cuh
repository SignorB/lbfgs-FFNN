#pragma once

#include "cublas_handle.cuh"
#include "network.cuh"

namespace cuda_mlp {

class CudaOptimizer {
public:
  explicit CudaOptimizer(CublasHandle &handle) : handle_(handle) {}
  virtual ~CudaOptimizer() = default;

  void setMaxIterations(int iters) { max_iters_ = iters; }
  void setTolerance(double tol) { tol_ = tol; }

  virtual void solve(CudaNetwork &net, const double *input, const double *target, int batch) = 0;

protected:
  CublasHandle &handle_;
  int max_iters_ = 200;
  double tol_ = 1e-6;
};

} // namespace cuda_mlp
