#pragma once

#include "../common.hpp"
#include "full_batch_minimizer.hpp"
#include <Eigen/Eigen>
#include <chrono>
#include <iostream>

namespace cpu_mlp {

/**
 * @brief Standard Gradient Descent with optional Line Search.
 * @details Uses the full gradient of the objective. Can perform a line search or use a fixed step.
 */
template <typename V, typename M> class GradientDescent : public FullBatchMinimizer<V, M> {
  // Inherit protected members
  using Base = FullBatchMinimizer<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  /**
   * @brief Sets the constant step size.
   * @param alpha Step size (effectively learning rate).
   */
  void setStepSize(double alpha) noexcept { step_size = alpha; }

  /**
   * @brief Enables or disables Line Search.
   * @param enable If true, uses Wolfe line search.
   */
  void useLineSearch(bool enable) noexcept { use_line_search = enable; }

  /**
   * @brief Run Gradient Descent.
   * @param x Initial guess.
   * @param f Objective function.
   * @param Gradient Gradient function.
   * @return Optimized parameters.
   */
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    const bool timing = (this->recorder_ != nullptr);
    if (this->recorder_) this->recorder_->reset();
    auto start_time = std::chrono::steady_clock::now();

    V g = Gradient(x);
    for (_iters = 0; _iters < _max_iters; ++_iters) {
      if (g.norm() < _tol) break;

      double alpha = step_size;
      if (use_line_search) alpha = this->line_search(x, -g, f, Gradient);

      x = x - alpha * g;
      g = Gradient(x);

      if (this->recorder_) {
        double loss = f(x);
        double elapsed_ms = 0.0;
        if (timing) {
          auto now = std::chrono::steady_clock::now();
          elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        }
        this->recorder_->record(_iters, loss, g.norm(), elapsed_ms);
      }
    }
    return x;
  }

private:
  double step_size = 1e-2;
  bool use_line_search = true;
};

} // namespace cpu_mlp
