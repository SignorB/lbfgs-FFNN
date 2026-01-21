#pragma once

#include "../common.hpp"
#include "../iteration_recorder.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <limits>

namespace cpu_mlp {

/**
 * @brief Base class for Full Batch Minimizers.
 * @details Implements common functionalities such as Line Search for deterministic optimization.
 */
template <typename V, typename M> class FullBatchMinimizer {
public:
  virtual ~FullBatchMinimizer() = default;

  /**
   * @brief Performs optimization.
   * @param x Initial guess.
   * @param f Objective function.
   * @param Gradient Gradient function.
   * @return Optimized parameter vector.
   */
  virtual V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) = 0;

  virtual void setInitialHessian(const M & /*hess*/) {}
  virtual void setHessian(const HessFun<V, M> & /*hessFun*/) {}

  /**
   * @brief Sets the maximum number of iterations.
   * @param max_iters Limit on iterations.
   */
  void setMaxIterations(int max_iters) { _max_iters = max_iters; }

  /**
   * @brief Sets the tolerance for convergence.
   * @param tol Tolerance value (gradient norm).
   */
  void setTolerance(double tol) { _tol = tol; }

  /**
   * @brief Returns the number of iterations performed.
   * @return Iteration count.
   */
  unsigned int iterations() const { return _iters; }

  /**
   * @brief Returns the tolerance used.
   * @return Tolerance value.
   */
  double tolerance() const { return _tol; }
  /**
   * @brief Attach a recorder for loss/grad history.
   * @param recorder Recorder instance (may be null).
   */
  void setRecorder(::IterationRecorder<CpuBackend> *recorder) { recorder_ = recorder; }

protected:
  unsigned int _max_iters = 1000;
  unsigned int _iters = 0;
  double _tol = 1e-10;
  ::IterationRecorder<CpuBackend> *recorder_ = nullptr; ///< Optional recorder for diagnostics

  // Line Search Parameters
  double c1 = 1e-4;
  double c2 = 0.9;
  double rho = 0.5;
  double max_line_iters = 50;

  /**
   * @brief Backtracking Line Search satisfying Wolfe Conditions.
   * @param x Current position.
   * @param p Search direction.
   * @param f Objective function.
   * @param Gradient Gradient function.
   * @return Step length alpha.
   */
  double line_search(V x, V p, VecFun<V, double> &f, GradFun<V> &Gradient) {
    double f_old = f(x);
    double grad_f_old = Gradient(x).dot(p);
    double inf = std::numeric_limits<double>::infinity();
    double alpha_min = 0.0;
    double alpha_max = inf;
    double alpha = 1.0;

    for (int i = 0; i < max_line_iters; ++i) {
      V x_new = x + alpha * p;
      double f_new = f(x_new);

      if (f_new > f_old + c1 * alpha * grad_f_old) {
        alpha_max = alpha;
        alpha = rho * (alpha_min + alpha_max);
        continue;
      }

      double grad_f_new_dot_p = Gradient(x_new).dot(p);

      if (grad_f_new_dot_p < c2 * grad_f_old) {
        alpha_min = alpha;
        if (alpha_max == inf)
          alpha *= 2;
        else
          alpha = rho * (alpha_min + alpha_max);
        continue;
      }
      return alpha;
    }
    return alpha;
  }
};

} // namespace cpu_mlp
