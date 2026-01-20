#pragma once

#include "../iteration_recorder.hpp"
#include "cublas_handle.cuh"
#include <algorithm>
#include <functional>
#include <vector>

namespace cuda_mlp {

/// @brief Abstract base class for CUDA-based minimizers
class CudaMinimizerBase {
public:
  /// @brief Loss and gradient callback signature
  using LossGradFun = std::function<CudaScalar(
      const CudaScalar *params, CudaScalar *grad, const CudaScalar *input, const CudaScalar *target, int batch)>;
  /// @brief Optional per-iteration hook signature
  using IterHook = std::function<void(int)>;

  /// @brief Construct with a cuBLAS handle reference
  explicit CudaMinimizerBase(CublasHandle &handle) : handle_(handle) {}
  virtual ~CudaMinimizerBase() = default;

  /// @brief Return the number of iterations performed in the last solve
  int iterations() const noexcept { return last_iterations_; }
  /// @brief Attach a recorder for loss/grad norm history
  void setRecorder(::IterationRecorder<CudaBackend> *recorder) { recorder_ = recorder; }

  /// @brief Set maximum number of iterations
  void setMaxIterations(int iters) { max_iters_ = iters; }
  /// @brief Set stopping tolerance (interpretation depends on optimizer)
  void setTolerance(CudaScalar tol) { tol_ = tol; }
  /**
   * @brief Configure Armijo line search parameters
   * @param max_iters Maximum line-search iterations
   * @param c1 Armijo sufficient decrease constant
   * @param rho Backtracking factor in (0,1)
   */
  void setLineSearchParams(int max_iters, CudaScalar c1, CudaScalar rho) {
    max_line_iters_ = (max_iters < 1) ? 1 : max_iters;
    c1_ = c1;
    rho_ = rho;
  }

  /**
   * @brief Solve the optimization problem
   * @param n Number of parameters
   * @param params Parameter vector (device pointer)
   * @param input Input data (device pointer)
   * @param target Target data (device pointer)
   * @param batch Batch size
   * @param loss_grad Callback that returns loss and writes gradient
   */
  virtual void solve(int n,
      CudaScalar *params,
      const CudaScalar *input,
      const CudaScalar *target,
      int batch,
      const LossGradFun &loss_grad) = 0;

protected:
  CublasHandle &handle_;                                 ///< cuBLAS handle used by the optimizer
  int max_iters_ = 200, max_line_iters_ = 20;            ///< Iteration limits
  CudaScalar tol_ = 1e-6f, c1_ = 1e-4f, rho_ = 0.5f;     ///< Stopping and line-search params
  int last_iterations_ = 0;                              ///< Iterations performed in last run
  ::IterationRecorder<CudaBackend> *recorder_ = nullptr; ///< Optional recorder for diagnostics
};

} // namespace cuda_mlp
