#pragma once

#include "common.cuh"
#include "cublas_handle.cuh"
#include "device_buffer.cuh"
#include <algorithm>
#include <functional>
#include <vector>

namespace cuda_mlp {

struct IterationRecorder {
public:
  void init(int capacity) {
    if (capacity <= 0) return;
    capacity_ = capacity;
    loss_.resize(static_cast<size_t>(capacity));
    grad_norm_.resize(static_cast<size_t>(capacity));
    size_ = 0;
  }

  void reset() { size_ = 0; }

  void record(int idx, CudaScalar loss, CudaScalar grad_norm) {
    if (idx < 0 || idx >= capacity_) return;
    cuda_check(cudaMemcpy(loss_.data() + idx, &loss, sizeof(CudaScalar), cudaMemcpyHostToDevice), "record loss");
    cuda_check(
        cudaMemcpy(grad_norm_.data() + idx, &grad_norm, sizeof(CudaScalar), cudaMemcpyHostToDevice), "record grad_norm");
    size_ = std::max(size_, idx + 1);
  }

  void copy_to_host(std::vector<CudaScalar> &loss_out, std::vector<CudaScalar> &grad_norm_out) const {
    loss_out.resize(size_);
    grad_norm_out.resize(size_);
    if (size_ == 0) return;
    loss_.copy_to_host(loss_out.data(), size_);
    grad_norm_.copy_to_host(grad_norm_out.data(), size_);
  }

  int size() const { return size_; }

private:
  DeviceBuffer<CudaScalar> loss_;
  DeviceBuffer<CudaScalar> grad_norm_;
  int capacity_ = 0;
  int size_ = 0;
};

class CudaMinimizerBase {
public:
  using LossGradFun = std::function<CudaScalar(
      const CudaScalar *params, CudaScalar *grad, const CudaScalar *input, const CudaScalar *target, int batch)>;
  using IterHook = std::function<void(int)>;

  explicit CudaMinimizerBase(CublasHandle &handle) : handle_(handle) {}
  virtual ~CudaMinimizerBase() = default;

  int iterations() const noexcept { return last_iterations_; }
  void setRecorder(IterationRecorder *recorder) { recorder_ = recorder; }

  void setMaxIterations(int iters) { max_iters_ = iters; }
  void setTolerance(CudaScalar tol) { tol_ = tol; }
  void setLineSearchParams(int max_iters, CudaScalar c1, CudaScalar rho) {
    max_line_iters_ = (max_iters < 1) ? 1 : max_iters;
    c1_ = c1;
    rho_ = rho;
  }

  virtual void solve(int n,
      CudaScalar *params,
      const CudaScalar *input,
      const CudaScalar *target,
      int batch,
      const LossGradFun &loss_grad) = 0;

protected:
  CublasHandle &handle_;
  int max_iters_ = 200, max_line_iters_ = 20;
  CudaScalar tol_ = 1e-6f, c1_ = 1e-4f, rho_ = 0.5f;
  int last_iterations_ = 0;
  IterationRecorder *recorder_ = nullptr;
};

} // namespace cuda_mlp
