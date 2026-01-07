#pragma once

#include "common.cuh"
#include "cublas_handle.cuh"
#include <functional>

namespace cuda_mlp {

class CudaMinimizerBase {
public:
  using LossGradFun = std::function<CudaScalar(const CudaScalar *params, CudaScalar *grad, const CudaScalar *input,
                                               const CudaScalar *target, int batch)>;

  explicit CudaMinimizerBase(CublasHandle &handle) : handle_(handle) {}
  virtual ~CudaMinimizerBase() = default;

  void setMaxIterations(int iters) { max_iters_ = iters; }
  void setTolerance(CudaScalar tol) { tol_ = tol; }
  void setLineSearchParams(int max_iters, CudaScalar c1, CudaScalar rho) {
    max_line_iters_ = (max_iters < 1) ? 1 : max_iters;
    c1_ = c1;
    rho_ = rho;
  }

  virtual void solve(int n, CudaScalar *params, const CudaScalar *input, const CudaScalar *target, int batch,
                     const LossGradFun &loss_grad) = 0;

protected:
  CublasHandle &handle_;
  int max_iters_ = 200, max_line_iters_ = 20;
  CudaScalar tol_ = 1e-6f, c1_ = 1e-4f, rho_ = 0.5f;
};

} // namespace cuda_mlp
