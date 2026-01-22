#pragma once

#include "../common.hpp"
#include "../iteration_recorder.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <functional>
#include <limits>

extern "C" {
extern double __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_out;
}

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

  /**
   * @brief Sets the initial Hessian approximation (if applicable).
   * @param hess Initial Hessian matrix.
   */
  virtual void setInitialHessian(const M & /*hess*/) {}

  /**
   * @brief Sets the Hessian function (for Second Order methods).
   * @param hessFun Function to compute the Hessian.
   */
  virtual void setHessian(const HessFun<V, M> & /*hessFun*/) {}

  /**
   * @brief Helper to solve directly using an Enzyme-compatible raw function.
   * @tparam LossFn The raw C++ function pointer for the loss .
   * @tparam DataType The type of the data structure passed to the loss.
   * @param x Initial parameter vector.
   * @param data Pointer to the data structure.
   * @return Optimized parameter vector.
   */
  template <auto LossFn, typename DataType> V solve_with_enzyme(V x, DataType *data) {
    VecFun<V, double> f = [data](const V &w) -> double { return LossFn(const_cast<double *>(w.data()), data); };
    GradFun<V> Gradient = [data](const V &w) -> V {
      V grad = V::Zero(w.size());
      __enzyme_autodiff((void *)LossFn, enzyme_dup, const_cast<double *>(w.data()), grad.data(), enzyme_const, data);
      return grad;
    };
    return this->solve(x, f, Gradient);
  }

  /**
   * @brief Sets the maximum number of iterations.
   * @param max_iters Limit on iterations.
   */
  void setMaxIterations(int max_iters) { _max_iters = max_iters; }
  /**
   * @brief Sets the maximum number of iterations for the line search.
   * @param max_line Maximum line search iterations.
   */
  void setMaxLineIters(int max_line) { max_line_iters = max_line; }

  /**
   * @brief Sets the maximum iterations for Armijo condition check (alias for setMaxLineIters).
   * @param max_armijo Maximum iterations.
   */
  void setArmijoMaxIter(int max_armijo) { max_line_iters = max_armijo; }

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
