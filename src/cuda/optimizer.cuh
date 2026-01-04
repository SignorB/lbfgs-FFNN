#pragma once

#include "cublas_handle.cuh"
#include "device_buffer.cuh"
#include "kernels.cuh"
#include "network.cuh"
#include <algorithm>
#include <vector>

namespace cuda_mlp {

class CudaLBFGS {
public:
  explicit CudaLBFGS(CublasHandle &handle) : handle_(handle) {}

  void setMaxIterations(int iters) { max_iters_ = iters; }
  void setTolerance(double tol) { tol_ = tol; }
  void setMemory(size_t m) { m_ = m; }

  void solve(CudaNetwork &net, const double *input, const double *target, int batch) {
    const int n = static_cast<int>(net.params_size());
    double *params = net.params_data();

    DeviceBuffer<double> grad(n);
    DeviceBuffer<double> grad_new(n);
    DeviceBuffer<double> p(n);
    DeviceBuffer<double> q(n);
    DeviceBuffer<double> z(n);
    DeviceBuffer<double> x_backup(n);

    std::vector<DeviceBuffer<double>> s_list;
    std::vector<DeviceBuffer<double>> y_list;
    std::vector<double> rho_list;
    s_list.reserve(m_);
    y_list.reserve(m_);
    rho_list.reserve(m_);

    double loss = net.compute_loss_and_grad(input, target, batch);
    device_copy(grad.data(), net.grads_data(), n);

    for (int iter = 0; iter < max_iters_; ++iter) {
      double grad_norm = device_nrm2(handle_, grad.data(), n);
      if (grad_norm < tol_) {
        break;
      }

      compute_direction(grad, s_list, y_list, rho_list, p, q, z);

      double grad_dot_p = device_dot(handle_, grad.data(), p.data(), n);
      double alpha = (iter == 0) ? std::min(1.0, 1.0 / grad_norm) : 1.0;

      device_copy(x_backup.data(), params, n);

      double loss_new = 0.0;
      bool accepted = false;
      for (int ls = 0; ls < max_line_iters_; ++ls) {
        device_copy(params, x_backup.data(), n);
        device_axpy(handle_, n, alpha, p.data(), params);

        loss_new = net.compute_loss_and_grad(input, target, batch);
        if (loss_new <= loss + c1_ * alpha * grad_dot_p) {
          device_copy(grad_new.data(), net.grads_data(), n);
          accepted = true;
          break;
        }
        alpha *= rho_;
      }
      if (!accepted) {
        device_copy(grad_new.data(), net.grads_data(), n);
      }

      DeviceBuffer<double> s(n);
      DeviceBuffer<double> y(n);
      device_copy(s.data(), params, n);
      device_axpy(handle_, n, -1.0, x_backup.data(), s.data());

      device_copy(y.data(), grad_new.data(), n);
      device_axpy(handle_, n, -1.0, grad.data(), y.data());

      double ys = device_dot(handle_, y.data(), s.data(), n);
      if (ys > 1e-10) {
        double rho = 1.0 / ys;
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

      std::cout << "Iter " << (iter + 1) << " - MSE: " << loss << std::endl;
    }
  }

private:
  void compute_direction(const DeviceBuffer<double> &grad, const std::vector<DeviceBuffer<double>> &s_list,
                         const std::vector<DeviceBuffer<double>> &y_list, const std::vector<double> &rho_list,
                         DeviceBuffer<double> &p, DeviceBuffer<double> &q, DeviceBuffer<double> &z) {
    const int n = static_cast<int>(grad.size());

    if (s_list.empty()) {
      device_copy(p.data(), grad.data(), n);
      device_scal(handle_, n, -1.0, p.data());
      return;
    }

    device_copy(q.data(), grad.data(), n);
    std::vector<double> alpha_list(s_list.size(), 0.0);

    for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i) {
      double alpha = rho_list[i] * device_dot(handle_, s_list[i].data(), q.data(), n);
      alpha_list[i] = alpha;
      device_axpy(handle_, n, -alpha, y_list[i].data(), q.data());
    }

    const DeviceBuffer<double> &s_last = s_list.back();
    const DeviceBuffer<double> &y_last = y_list.back();
    double ys = device_dot(handle_, s_last.data(), y_last.data(), n);
    double yy = device_dot(handle_, y_last.data(), y_last.data(), n);
    double gamma = (yy > 0.0) ? (ys / yy) : 1.0;

    device_copy(z.data(), q.data(), n);
    device_scal(handle_, n, gamma, z.data());

    for (size_t i = 0; i < s_list.size(); ++i) {
      double beta = rho_list[i] * device_dot(handle_, y_list[i].data(), z.data(), n);
      double scale = alpha_list[i] - beta;
      device_axpy(handle_, n, scale, s_list[i].data(), z.data());
    }

    device_copy(p.data(), z.data(), n);
    device_scal(handle_, n, -1.0, p.data());
  }

  CublasHandle &handle_;
  int max_iters_ = 200;
  int max_line_iters_ = 20;
  double tol_ = 1e-6;
  size_t m_ = 16;
  double c1_ = 1e-4;
  double rho_ = 0.5;
};

} // namespace cuda_mlp
