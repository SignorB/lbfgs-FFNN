#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace cuda_mlp {

class CudaLBFGS : public CudaMinimizerBase {
public:
  explicit CudaLBFGS(CublasHandle &handle) : CudaMinimizerBase(handle) {}
  void setMemory(size_t m) { m_ = m; }

  void solve(int n,
      CudaScalar *params,
      const CudaScalar *input,
      const CudaScalar *target,
      int batch,
      const LossGradFun &loss_grad) override {
    if (n <= 0 || params == nullptr) {
      last_iterations_ = 0;
      return;
    }

    last_iterations_ = 0;
    if (recorder_) recorder_->reset();

    DeviceBuffer<CudaScalar> grad(n), grad_new(n), p(n), q(n), z(n), x_backup(n);
    const int m = static_cast<int>(m_);
    std::vector<DeviceBuffer<CudaScalar>> s_hist;
    std::vector<DeviceBuffer<CudaScalar>> y_hist;
    std::vector<CudaScalar> rho_hist;

    int hist_head = 0;
    int hist_count = 0;

    if (m > 0) {
      s_hist.resize(m);
      y_hist.resize(m);
      rho_hist.assign(m, CudaScalar{0});

      for (int i = 0; i < m; ++i) {
        s_hist[i].resize(static_cast<size_t>(n));
        y_hist[i].resize(static_cast<size_t>(n));
      }
    }

    auto reset_history = [&]() {
      hist_head = 0;
      hist_count = 0;
    };

    CudaScalar loss = loss_grad(params, grad.data(), input, target, batch);

    int iterations_done = 0;
    for (int iter = 0; iter < max_iters_; ++iter) {
      CudaScalar grad_norm = device_nrm2(handle_, grad.data(), n);
      if (grad_norm < tol_) break;

      compute_direction_ring(grad, s_hist, y_hist, rho_hist, hist_head, hist_count, p, q, z);
      CudaScalar grad_dot_p = device_dot(handle_, grad.data(), p.data(), n);
      if (grad_dot_p >= CudaScalar{0}) {
        device_copy(p.data(), grad.data(), n);
        device_scal(handle_, n, CudaScalar{-1}, p.data());
        grad_dot_p = -device_dot(handle_, grad.data(), grad.data(), n);
        reset_history();
      }

      CudaScalar alpha = (iter == 0) ? std::min(CudaScalar{1}, CudaScalar{1} / grad_norm) : CudaScalar{1};
      device_copy(x_backup.data(), params, n);

      CudaScalar loss_new = CudaScalar{0};
      bool evaluated = false;
      bool armijo_ok = false;

      for (int ls = 0; ls < max_line_iters_; ++ls) {
        device_copy(params, x_backup.data(), n);
        device_axpy(handle_, n, alpha, p.data(), params);

        loss_new = loss_grad(params, grad_new.data(), input, target, batch);
        evaluated = true;

        if (loss_new <= loss + c1_ * alpha * grad_dot_p) {
          armijo_ok = true;
          break;
        }

        CudaScalar denominator = CudaScalar{2} * (loss_new - loss - grad_dot_p * alpha);
        bool use_fallback = true;

        if (std::abs(denominator) > CudaScalar{1e-20}) {
          CudaScalar new_alpha = -(grad_dot_p * alpha * alpha) / denominator;
          if (new_alpha >= CudaScalar{0.1} * alpha && new_alpha <= CudaScalar{0.9} * alpha) {
            alpha = new_alpha;
            use_fallback = false;
          }
        }

        if (use_fallback) alpha *= rho_;
      }

      if (!evaluated) {
        loss_new = loss_grad(params, grad_new.data(), input, target, batch);
        armijo_ok = true;
      }

      if (!armijo_ok) reset_history();

      if (m > 0) {
        const int slot = hist_head;

        // s = params - x_backup
        device_copy(s_hist[slot].data(), params, n);
        device_axpy(handle_, n, CudaScalar{-1}, x_backup.data(), s_hist[slot].data());

        // y = grad_new - grad
        device_copy(y_hist[slot].data(), grad_new.data(), n);
        device_axpy(handle_, n, CudaScalar{-1}, grad.data(), y_hist[slot].data());

        CudaScalar ys = device_dot(handle_, y_hist[slot].data(), s_hist[slot].data(), n);

        if (ys > CudaScalar{1e-10}) {
          rho_hist[slot] = CudaScalar{1} / ys;

          hist_head = (hist_head + 1) % m;
          hist_count = std::min(hist_count + 1, m);
        }
      }


      device_copy(grad.data(), grad_new.data(), n);
      loss = loss_new;

      CudaScalar grad_norm_new = device_nrm2(handle_, grad.data(), n);
      if (recorder_) recorder_->record(iterations_done, loss, grad_norm_new);

      iterations_done++;
    }

    last_iterations_ = iterations_done;
  }

private:
  void compute_direction_ring(const DeviceBuffer<CudaScalar> &grad,
      const std::vector<DeviceBuffer<CudaScalar>> &s_hist,
      const std::vector<DeviceBuffer<CudaScalar>> &y_hist,
      const std::vector<CudaScalar> &rho_hist,
      int hist_head,
      int hist_count,
      DeviceBuffer<CudaScalar> &p,
      DeviceBuffer<CudaScalar> &q,
      DeviceBuffer<CudaScalar> &z) {
    const int n = static_cast<int>(grad.size());

    if (hist_count <= 0 || s_hist.empty()) {
      device_copy(p.data(), grad.data(), n);
      device_scal(handle_, n, CudaScalar{-1}, p.data());
      return;
    }

    const int m = static_cast<int>(s_hist.size());
    auto phys = [&](int logical_idx) -> int {
      int start = (hist_head - hist_count);
      start %= m;
      if (start < 0) start += m;
      return (start + logical_idx) % m;
    };

    device_copy(q.data(), grad.data(), n);

    std::vector<CudaScalar> alpha(static_cast<size_t>(hist_count), CudaScalar{0});

    for (int li = hist_count - 1; li >= 0; --li) {
      const int i = phys(li);
      CudaScalar a = rho_hist[i] * device_dot(handle_, s_hist[i].data(), q.data(), n);
      alpha[static_cast<size_t>(li)] = a;
      device_axpy(handle_, n, -a, y_hist[i].data(), q.data());
    }

    const int last = phys(hist_count - 1);
    CudaScalar ys = device_dot(handle_, s_hist[last].data(), y_hist[last].data(), n);
    CudaScalar yy = device_dot(handle_, y_hist[last].data(), y_hist[last].data(), n);
    CudaScalar gamma = (yy > CudaScalar{0}) ? (ys / yy) : CudaScalar{1};

    device_copy(z.data(), q.data(), n);
    device_scal(handle_, n, gamma, z.data());

    for (int li = 0; li < hist_count; ++li) {
      const int i = phys(li);
      CudaScalar b = rho_hist[i] * device_dot(handle_, y_hist[i].data(), z.data(), n);
      CudaScalar scale = alpha[static_cast<size_t>(li)] - b;
      device_axpy(handle_, n, scale, s_hist[i].data(), z.data());
    }

    device_copy(p.data(), z.data(), n);
    device_scal(handle_, n, CudaScalar{-1}, p.data());
  }

  size_t m_ = 16;
};

} // namespace cuda_mlp
