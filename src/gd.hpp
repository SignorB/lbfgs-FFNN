#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <Eigen/Eigen>
#include <iostream>

/**
 * @brief Full-batch gradient descent with optional line search.
 *
 * Uses the gradient of the full objective and, by default, a line search to
 * pick the step length. You can disable the line search to use a fixed step.
 */
template <typename V, typename M>
class GradientDescent : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  void setStepSize(double alpha) noexcept { step_size = alpha; }
  void useLineSearch(bool enable) noexcept { use_line_search = enable; }

  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
        
    for (_iters = 0; _iters < _max_iters; ++_iters) {
      std::cout << "iter: " << _iters << std::endl;

      V g = Gradient(x);
      if (g.norm() < _tol)
        break;

      double alpha = step_size;
      if (use_line_search)
        alpha = this->line_search(x, -g, f, Gradient);

      x = x - alpha * g;
    }
    return x;
  }

  using Base::solve;

private:
  double step_size = 1e-2;
  bool use_line_search = true;
};
