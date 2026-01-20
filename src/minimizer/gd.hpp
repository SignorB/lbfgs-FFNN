#pragma once

#include "../common.hpp"
#include "minimizer_base.hpp"
#include <Eigen/Eigen>
#include <chrono>
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
    const bool timing = (this->recorder_ != nullptr);
    if (this->recorder_) this->recorder_->reset();
    auto start_time = std::chrono::steady_clock::now();

    for (_iters = 0; _iters < _max_iters; ++_iters) {
      V g = Gradient(x);
      if (g.norm() < _tol)
        break;

      double alpha = step_size;
      if (use_line_search)
        alpha = this->line_search(x, -g, f, Gradient);

      x = x - alpha * g;

      if (this->recorder_) {
        double loss = f(x);
        double elapsed_ms = 0.0;
        if (timing) {
          auto now = std::chrono::steady_clock::now();
          elapsed_ms =
              std::chrono::duration<double, std::milli>(now - start_time).count();
        }
        this->recorder_->record(_iters, loss, g.norm(), elapsed_ms);
      }
    }
    return x;
  }

  using Base::solve;

private:
  double step_size = 1e-2;
  bool use_line_search = true;
};
