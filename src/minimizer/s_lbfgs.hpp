#pragma once

#include "../common.hpp"
#include "ring_buffer.hpp"
#include "stochastic_minimizer.hpp"

#include <Eigen/Eigen>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>

namespace cpu_mlp {

/**
 * @brief Stochastic Limited-memory BFGS (S-LBFGS) minimizer.
 * @details Implements a stochastic variance-reduced quasi-Newton method with batch callbacks.
 */
template <typename V, typename M> class SLBFGS : public StochasticMinimizer<V, M> {
public:
  using Base = StochasticMinimizer<V, M>;

protected:
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
  using Base::step_size;

public:
  using Base::setMaxIterations;
  using Base::setStepSize;
  using Base::setTolerance;

  using BatchGradFun = std::function<void(const V &, const std::vector<size_t> &, V &)>;
  using BatchLossFun = std::function<double(const V &, const std::vector<size_t> &)>;

  /**
   * @brief Stochastic Solve using Batch Callbacks.
   * @param weights Initial parameter weights.
   * @param f Loss function callback (evaluates loss on batch indices).
   * @param batch_g Gradient function callback (computes gradient on batch indices).
   * @param m Number of stochastic steps per epoch.
   * @param M_param History size for L-BFGS pairs.
   * @param L Hessian update interval.
   * @param b Batch size for gradient steps.
   * @param b_H Batch size for Hessian vector products.
   * @param step_size Learning rate.
   * @param N Total dataset size (used for indexing).
   * @return Optimized weights.
   */
  V stochastic_solve(V weights,
      const BatchLossFun &f,
      const BatchGradFun &batch_g,
      int m,
      int M_param,
      int L,
      int b,
      int b_H,
      double step_size,
      int N);

  /**
   * @brief Helper to sample minibatch indices.
   */
  static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);

  void setData(const BatchLossFun &f, const BatchGradFun &g) {
    _sf = f;
    _sg = g;
  }

private:
  BatchLossFun _sf;
  BatchGradFun _sg;
};

// -------------------------------------------------------------------------
// Helper: Finite Difference HVP on a Batch
// -------------------------------------------------------------------------
template <typename V, typename BatchFn>
V finite_difference_hvp_batch(
    BatchFn &g, const V &weights, const std::vector<size_t> &indices, const V &v, double epsilon = 1e-4) {
  V w_plus = weights + epsilon * v;
  V w_minus = weights - epsilon * v;

  V grad_plus = V::Zero(weights.size());
  V grad_minus = V::Zero(weights.size());

  g(w_plus, indices, grad_plus);
  g(w_minus, indices, grad_minus);

  return (grad_plus - grad_minus) / (2.0 * epsilon);
}

// -------------------------------------------------------------------------
// Helper: L-BFGS Two Loop Recursion
// -------------------------------------------------------------------------
template <typename V>
V lbfgs_two_loop(const RingBuffer<V> &s_list, const RingBuffer<V> &y_list, const RingBuffer<double> &rho_list, const V &v) {
  int M = s_list.size();
  std::vector<double> alpha(M);
  V q = v;

  // Backward
  for (int i = M - 1; i >= 0; --i) {
    alpha[i] = rho_list[i] * s_list[i].dot(q);
    q = q - alpha[i] * y_list[i];
  }

  // Scaling
  double gamma = 1.0;
  if (M > 0) {
    double denom = y_list.back().dot(y_list.back());
    if (std::abs(denom) < 1e-12)
      gamma = 1.0;
    else
      gamma = s_list.back().dot(y_list.back()) / denom;
    gamma = std::min(std::max(gamma, 1e-6), 1e6);
  }
  V r = gamma * q;

  // Forward
  for (int i = 0; i < M; ++i) {
    double beta = rho_list[i] * y_list[i].dot(r);
    r = r + s_list[i] * (alpha[i] - beta);
  }
  return r;
}

// -------------------------------------------------------------------------
// Implementation: sample_minibatch_indices
// -------------------------------------------------------------------------
template <typename V, typename M>
std::vector<size_t> SLBFGS<V, M>::sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng) {
  if (N == 0 || batch_size == 0) return {};
  if (batch_size >= N) {
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    return idx;
  }

  // Partial Fisher-Yates
  std::vector<size_t> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  for (size_t i = 0; i < batch_size; ++i) {
    std::uniform_int_distribution<size_t> dist(i, N - 1);
    size_t j = dist(rng);
    std::swap(idx[i], idx[j]);
  }
  idx.resize(batch_size);
  return idx;
}

// -------------------------------------------------------------------------
// Implementation: stochastic_solve
// -------------------------------------------------------------------------
template <typename V, typename M>
V SLBFGS<V, M>::stochastic_solve(V weights,
    const BatchLossFun &f,
    const BatchGradFun &batch_g,
    int m,
    int M_param,
    int L,
    int b,
    int b_H,
    double step_size,
    int N) {

  _iters = 0;
  RingBuffer<V> u_list(M_param > 0 ? M_param + 1 : 0);
  RingBuffer<V> s_list(M_param > 0 ? M_param : 0);
  RingBuffer<V> y_list(M_param > 0 ? M_param : 0);
  RingBuffer<double> rho_list(M_param > 0 ? M_param : 0);

  int seed = 56;
  std::mt19937 rng(seed);

  int dim_weights = weights.size();
  V wt = weights;
  this->step_size = step_size;

  RingBuffer<V> w_history(L + 1);

  // Full Batch Indices
  std::vector<size_t> full_indices(N);
  std::iota(full_indices.begin(), full_indices.end(), 0);

  const bool timing = (this->recorder_ != nullptr);
  if (this->recorder_) this->recorder_->reset();
  auto start_time = std::chrono::steady_clock::now();

  while (_iters < _max_iters) {

    w_history.clear();

    // 1. Compute Full Gradient (Variance Reduction Anchor)
    V full_gradient = V::Zero(dim_weights);

    batch_g(weights, full_indices, full_gradient);

    if (full_gradient.norm() < _tol) {
      std::cout << "Converged: gradient norm " << full_gradient.norm() << std::endl;
      break;
    }

    wt = weights;
    w_history.push_back(wt);
    V variance_reduced_gradient = V::Zero(dim_weights);

    // 2. Inner Loop (Stochastic Updates)
    for (int t = 0; t < m; ++t) {

      auto minibatch_indices = sample_minibatch_indices(N, b, rng);

      V grad_estimate_wt = V::Zero(dim_weights);
      V grad_estimate_wk = V::Zero(dim_weights);

      batch_g(wt, minibatch_indices, grad_estimate_wt);
      batch_g(weights, minibatch_indices, grad_estimate_wk);

      variance_reduced_gradient = (grad_estimate_wt - grad_estimate_wk) + full_gradient;

      V direction = lbfgs_two_loop(s_list, y_list, rho_list, variance_reduced_gradient);
      wt = wt - this->step_size * direction;

      w_history.push_back(wt);

      // 3. Hessian Update (Curvature Pairs)
      if (t > 0 && t % L == 0) {

        V u = V::Zero(dim_weights);
        const int num_wt = static_cast<int>(w_history.size());
        for (size_t i = 0; i < w_history.size(); ++i)
          u += w_history[i];
        if (num_wt > 0) u /= static_cast<double>(num_wt);

        if (!u_list.empty()) {
          const V &u_prev = u_list.back();
          V s = u - u_prev;

          auto batch_indices_H = sample_minibatch_indices(N, b_H, rng);

          V y = finite_difference_hvp_batch(batch_g, u, batch_indices_H, s);

          double ys = y.dot(s);
          if (std::abs(ys) > 1e-10) {
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(1.0 / ys);
          }
        }

        u_list.push_back(u);
      }
    }

    // Reset anchor
    if (w_history.size() >= 2) {
      std::uniform_int_distribution<size_t> pick_i(0, w_history.size() - 2);
      weights = w_history[pick_i(rng)];
    } else {
      weights = wt;
    }

    // Logging (approximate loss via simple full gradient norm or callback if desired)
    // Calculating full loss is expensive. We skip it or use passed callback on full batch.
    if (this->recorder_) {
      double full_loss = f(weights, full_indices);
      V grad_log = V::Zero(dim_weights);
      batch_g(weights, full_indices, grad_log);
      double elapsed_ms = 0.0;
      if (timing) {
        auto now = std::chrono::steady_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
      }
      this->recorder_->record(_iters, full_loss, grad_log.norm(), elapsed_ms);
    }

    _iters++;
  }

  return weights;
};
} // namespace cpu_mlp
