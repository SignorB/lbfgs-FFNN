#pragma once

#include "../common.hpp"
#include "full_batch_minimizer.hpp"
#include <Eigen/Eigen>
#include <iostream>

namespace cpu_mlp {

/**
 * @brief Newton minimizer (full Newton) for unconstrained optimization.
 */
template <typename V, typename M>
class Newton : public FullBatchMinimizer<V, M> {
  using Base = FullBatchMinimizer<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
  using Base::line_search;

public:
  void setHessian(const HessFun<V, M> &hessFun) noexcept { _hessFun = hessFun; }

  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    Eigen::LDLT<M> ldlt;

    check(static_cast<bool>(_hessFun), "Hessian function must be set for Newton solver");

    for (_iters = 0; _iters < _max_iters; ++_iters) {
      V g = Gradient(x);
      double gnorm = g.norm();
      if (gnorm <= _tol)
        break;

      M H = _hessFun(x);
      check(H.rows() == H.cols(), "Hessian must be square");
      check(H.rows() == g.size(), "Hessian/gradient size mismatch");

      V p;
      bool found = false;
      double mu = reg_init;
      const double max_mu = reg_max;

      while (mu <= max_mu) {
        M Hreg = H;
        Hreg += mu * M::Identity(H.rows(), H.cols());

        ldlt.compute(Hreg);
        if (ldlt.info() == Eigen::Success) {
          p = ldlt.solve(-g);
          if (ldlt.info() == Eigen::Success && p.dot(g) < 0.0) {
            found = true;
            break;
          }
        }
        mu *= reg_growth;
      }

      if (!found) {
        p = -g;
      }

      double alpha = line_search(x, p, f, Gradient);
      x = x + alpha * p;
    }

    return x;
  }

private:
  HessFun<V, M> _hessFun;
  double reg_init = 1e-6;   // Initial diagonal damping.
  double reg_max = 1e6;     // Maximum damping.
  double reg_growth = 10.0; // Growth factor for damping.
};

} // namespace cpu_mlp
