#pragma once

#include "device_buffer.cuh"
#include "kernels.cuh"
#include "minimizer_base.cuh"
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

namespace cuda_mlp {

class CudaLBFGS : public CudaMinimizerBase {
public:
  explicit CudaLBFGS(CublasHandle &handle) : CudaMinimizerBase(handle) {}
  void setMemory(size_t m) { m_ = m; }

  void solve(int n, CudaScalar *params, const CudaScalar *input, const CudaScalar *target, int batch,
             const LossGradFun &loss_grad) override {
    if (n <= 0 || params == nullptr) {
      return;
    }

    DeviceBuffer<CudaScalar> grad(n), grad_new(n), p(n), q(n), z(n), x_backup(n);
    std::vector<CudaScalar> rho_list;
    std::vector<DeviceBuffer<CudaScalar>> s_list, y_list;

    s_list.reserve(m_);
    y_list.reserve(m_);
    rho_list.reserve(m_);

    CudaScalar loss = loss_grad(params, grad.data(), input, target, batch);

    for (int iter = 0; iter < max_iters_; ++iter) {
      CudaScalar grad_norm = device_nrm2(handle_, grad.data(), n);
      if (grad_norm < tol_) {
        break;
      }

      compute_direction(grad, s_list, y_list, rho_list, p, q, z);

      CudaScalar grad_dot_p = device_dot(handle_, grad.data(), p.data(), n);
      CudaScalar alpha = (iter == 0) ? std::min(CudaScalar{1.0f}, CudaScalar{1.0f} / grad_norm) : CudaScalar{1.0f};

      device_copy(x_backup.data(), params, n);

      CudaScalar loss_new = 0.0f;
      bool evaluated = false;
      for (int ls = 0; ls < max_line_iters_; ++ls) {
        device_copy(params, x_backup.data(), n);
        device_axpy(handle_, n, alpha, p.data(), params);

        loss_new = loss_grad(params, grad_new.data(), input, target, batch);
        evaluated = true;
        if (loss_new <= loss + c1_ * alpha * grad_dot_p) {
          break;
        }
        alpha *= rho_;
      }
      if (!evaluated) {
        loss_new = loss_grad(params, grad_new.data(), input, target, batch);
      }

      DeviceBuffer<CudaScalar> s(n), y(n);
      device_copy(s.data(), params, n);
      device_axpy(handle_, n, -1.0f, x_backup.data(), s.data());

      device_copy(y.data(), grad_new.data(), n);
      device_axpy(handle_, n, -1.0f, grad.data(), y.data());

      CudaScalar ys = device_dot(handle_, y.data(), s.data(), n);
      if (m_ > 0 && ys > 1e-10f) {
        CudaScalar rho = 1.0f / ys;
        if (s_list.size() == m_) {
          s_list.erase(s_list.begin());
          y_list.erase(y_list.begin());
          rho_list.erase(rho_list.begin());
        }
        s_list.emplace_back(std::move(s));
        y_list.emplace_back(std::move(y));
        rho_list.push_back(rho);
      }

      device_copy(grad.data(), grad_new.data(), n);
      loss = loss_new;

      std::cout << "Iter " << (iter + 1) << " - loss: " << loss << std::endl;
    }
  }

private:
  void compute_direction(const DeviceBuffer<CudaScalar> &grad, const std::vector<DeviceBuffer<CudaScalar>> &s_list,
                         const std::vector<DeviceBuffer<CudaScalar>> &y_list, const std::vector<CudaScalar> &rho_list,
                         DeviceBuffer<CudaScalar> &p, DeviceBuffer<CudaScalar> &q, DeviceBuffer<CudaScalar> &z) {
    const int n = static_cast<int>(grad.size());

    if (s_list.empty()) {
      device_copy(p.data(), grad.data(), n);
      device_scal(handle_, n, -1.0f, p.data());
      return;
    }

    device_copy(q.data(), grad.data(), n);
    std::vector<CudaScalar> alpha_list(s_list.size(), 0.0f);

    for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i) {
      CudaScalar alpha = rho_list[i] * device_dot(handle_, s_list[i].data(), q.data(), n);
      alpha_list[i] = alpha;
      device_axpy(handle_, n, -alpha, y_list[i].data(), q.data());
    }

    const DeviceBuffer<CudaScalar> &s_last = s_list.back(), &y_last = y_list.back();

    CudaScalar ys = device_dot(handle_, s_last.data(), y_last.data(), n),
               yy = device_dot(handle_, y_last.data(), y_last.data(), n), gamma = (yy > 0.0f) ? (ys / yy) : 1.0f;

    device_copy(z.data(), q.data(), n);
    device_scal(handle_, n, gamma, z.data());

    for (size_t i = 0; i < s_list.size(); ++i) {
      CudaScalar beta = rho_list[i] * device_dot(handle_, y_list[i].data(), z.data(), n);
      CudaScalar scale = alpha_list[i] - beta;
      device_axpy(handle_, n, scale, s_list[i].data(), z.data());
    }

    device_copy(p.data(), z.data(), n);
    device_scal(handle_, n, -1.0f, p.data());
  }

  size_t m_ = 16;
};

} // namespace cuda_mlp
