#pragma once

#include "../common.hpp"
#include "full_batch_minimizer.hpp"
#include <Eigen/Eigen>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

namespace cpu_mlp {

template <typename M> constexpr bool isSparse = std::is_base_of_v<Eigen::SparseMatrixBase<M>, M>;

template <typename M>
using DefaultSolverT = typename std::conditional<isSparse<M>, Eigen::ConjugateGradient<M>, Eigen::LDLT<M>>::type;

/**
 * @brief BFGS (Broyden–Fletcher–Goldfarb–Shanno) minimizer.
 */
template <typename V, typename M, typename Solver = DefaultSolverT<M>> class BFGS : public FullBatchMinimizer<V, M> {
  using Base = FullBatchMinimizer<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

protected:
  static constexpr bool UseDefaultSolver = std::is_same_v<Solver, DefaultSolverT<M>>;
  using SolverT = typename std::conditional<UseDefaultSolver, Solver, Solver &>::type;

private:
  SolverT _solver;
  M _B;

public:
  BFGS()
  requires(UseDefaultSolver) { _solver = DefaultSolverT<M>(); }

  BFGS(Solver &solver)
  requires(!UseDefaultSolver) : _solver(solver) {}

  /**
   * @brief Sets the initial approximate Hessian matrix.
   * @param b Initial Hessian approximation (usually Identity).
   */
  void setInitialHessian(const M &b) { _B = b; }

  /**
   * @brief Solves the optimization problem using BFGS method.
   * @param x Initial guess for the parameters.
   * @param f Objective function to minimize.
   * @param Gradient Function to compute the gradient of f.
   * @return Optimized parameter vector.
   */
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {

    // Initialize B if empty/size mismatch?
    // Usually B0 = I.
    if (_B.rows() != x.size()) {
      _B = M::Identity(x.size(), x.size());
    }

    for (_iters = 0; _iters < _max_iters && Gradient(x).norm() > _tol; ++_iters) {

      _solver.compute(_B);
      check(_solver.info() == Eigen::Success, "conjugate gradient solver error");

      V p = _solver.solve(-Gradient(x));

      double alpha = 1.0;
      alpha = this->line_search(x, p, f, Gradient);

      V s = alpha * p;
      V x_next = x + s;

      V y = Gradient(x_next) - Gradient(x);

      M b_prod = _B * s;
      _B = _B + (y * y.transpose()) / (y.transpose() * s) - (b_prod * b_prod.transpose()) / (s.transpose() * _B * s);

      x = x_next;
    }

    return x;
  }
};

} // namespace cpu_mlp
