#pragma once

#include "../common.hpp"
#include "full_batch_minimizer.hpp"
#include "ring_buffer.hpp"
#include <autodiff/forward/dual.hpp>
#include <Eigen/Eigen>

namespace cpu_mlp {

/**
 * @brief Limited-memory BFGS (L-BFGS) minimizer.
 * @details Approximates the inverse Hessian using a history of size m.
 */
template <typename V, typename M>
class LBFGS : public FullBatchMinimizer<V, M> {
  using Base = FullBatchMinimizer<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  /**
   * @brief Set history size for L-BFGS.
   * @param history_size Number of curvature pairs to store (m).
   */
  void setHistorySize(size_t history_size) { m = history_size; }

  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    RingBuffer<V> s_list(m);        
    RingBuffer<V> y_list(m);        
    RingBuffer<double> rho_list(m); 

    V grad = Gradient(x); 
    V p = -grad;          
    V x_new = x;          

    for (_iters = 0; _iters < _max_iters; ++_iters) {
      if (grad.norm() < _tol) {
        break;
      }

      p = compute_direction(grad, s_list, y_list, rho_list);

      double alpha_wolfe;
      // Heuristic for the first step
      if (_iters == 0)
        alpha_wolfe = std::min(1.0, 1.0 / grad.norm());
      else
        alpha_wolfe = this->line_search(x, p, f, Gradient);

      x_new = x + alpha_wolfe * p;
      V s = x_new - x;

      V grad_new = Gradient(x_new);
      V y = grad_new - grad;

      x = x_new;

      double ys = y.dot(s); 

      if (ys > 1e-10) { 
          double rho = 1.0 / ys;
          s_list.push_back(s);
          y_list.push_back(y);
          rho_list.push_back(rho);


      }
      
      grad = grad_new;
    }

    return x;
  }

  /**
   * @brief Two-loop recursion to compute search direction.
   */
  V compute_direction(const V &grad,
                      const RingBuffer<V> &s_list,
                      const RingBuffer<V> &y_list,
                      const RingBuffer<double> &rho_list) {

    if (s_list.empty()) {
      return -grad;
    }

    V z = V::Zero(grad.size());
    V q = grad;
    std::vector<double> alpha_list(s_list.size());

    // Backward pass
    for (int i = static_cast<int>(s_list.size()) - 1; i >= 0; --i) {
      alpha_list[i] = rho_list[i] * s_list[i].dot(q);
      q -= alpha_list[i] * y_list[i];
    }

    // Scaling
    double gamma = s_list.back().dot(y_list.back()) /
                   y_list.back().dot(y_list.back());

    z = gamma * q;

    // Forward pass
    for (size_t i = 0; i < s_list.size(); ++i) {
      double beta = rho_list[i] * y_list[i].dot(z);
      z += s_list[i] * (alpha_list[i] - beta);
    }

    return -z;
  }

private:
  size_t m = 16; 
};

} // namespace cpu_mlp
