#pragma once

#include "../common.hpp"
#include "stochastic_minimizer.hpp"
#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <random>

namespace cpu_mlp {

/**
 * @brief Stochastic Gradient Descent (SGD) minimizer.
 * @details Operates on minibatches provided by callbacks.
 */
template <typename V, typename M> class StochasticGradientDescent : public StochasticMinimizer<V, M> {
  // Public inheritance needed!
  using Base = StochasticMinimizer<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
  using Base::step_size;

public:
  using Base::setMaxIterations;
  using Base::setStepSize;
  using Base::setTolerance;

  using S_VecFun = std::function<double(const V &, const V &, const V &)>;
  using BatchGradFun = std::function<void(const V &, const std::vector<size_t> &, V &)>;

  StochasticGradientDescent() = default;

  /**
   * @brief Helper to sample minibatch indices without replacement (Fisher-Yates shuffle subset).
   */
  static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);

  /**
   * @brief Configure data and callbacks for the solver.
   */
  void setData(const M &inputs, const M &targets, const S_VecFun &f, const BatchGradFun &g) {
    _inputs = inputs;
    _targets = targets;
    _sf = f;
    _batch_g = g;
  }

  /**
   * @brief Execute Stochastic Solve.
   * @param init_w Initial weights.
   * @param m Number of minibatches per epoch.
   * @param b Batch size.
   * @param step Step size/learning rate.
   * @param verbose Print progress.
   * @param print_every Print interval.
   * @return Optimized weights.
   */
  V stochastic_solve(const V &init_w,
      int m,
      int b,
      double step,
      bool verbose = false,

      int print_every = 50) {
    if (_inputs.cols() == 0 || _targets.cols() == 0) {
      throw std::runtime_error("No data set for StochasticGradientDescent::stochastic_solve");
    }

    int N = static_cast<int>(_inputs.cols());
    std::vector<size_t> full_indices(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
      full_indices[static_cast<size_t>(i)] = static_cast<size_t>(i);
    }

    V w = init_w;
    int dim = static_cast<int>(w.size());
    step_size = step;

    double passes = 0.0;
    std::mt19937 rng(123);

    const bool timing = (this->recorder_ != nullptr);
    if (this->recorder_) this->recorder_->reset();
    auto start_time = std::chrono::steady_clock::now();

    while (_iters < _max_iters) {
      // one epoch of m minibatches
      double last_grad_norm = 0.0;

      for (int t = 0; t < m; ++t) {
        auto minibatch_indices = sample_minibatch_indices(N, b, rng);

        V grad_est = V::Zero(dim);

        _batch_g(w, minibatch_indices, grad_est);
        last_grad_norm = grad_est.norm();

        w = w - step_size * grad_est;

        passes += static_cast<double>(b) / static_cast<double>(N);
      }

      double loss = 0.0;
      for (int i = 0; i < N; ++i) {
        loss += _sf(w, _inputs.col(i), _targets.col(i));
      }
      loss /= static_cast<double>(N);

      if (verbose && ((_iters % print_every) == 0)) {
        std::cout << "Epoch " << (_iters + 1) << " loss=" << loss << " passes=" << passes << std::endl;
      }

      if (this->recorder_) {
        double grad_norm = last_grad_norm;
        if (N > 0) {
          V full_grad = V::Zero(dim);
          _batch_g(w, full_indices, full_grad);
          grad_norm = full_grad.norm();
        }
        double elapsed_ms = 0.0;
        if (timing) {
          auto now = std::chrono::steady_clock::now();
          elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        }
        this->recorder_->record(_iters, loss, grad_norm, elapsed_ms);
      }

      _iters++;
    }

    return w;
  }

private:
  M _inputs;
  M _targets;
  S_VecFun _sf;
  BatchGradFun _batch_g;
};

template <typename V, typename M>
std::vector<size_t> StochasticGradientDescent<V, M>::sample_minibatch_indices(
    const size_t N, size_t batch_size, std::mt19937 &rng) {
  if (N == 0 || batch_size == 0) {
    std::cerr << "Warning: sampling minibatch from empty dataset or with batch_size=0." << std::endl;
    return {};
  }

  std::vector<size_t> idx(N);
  for (size_t i = 0; i < N; ++i) {
    idx[i] = i;
  }

  if (batch_size >= N) {
    return idx;
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::uniform_int_distribution<size_t> dist(i, N - 1);
    size_t j = dist(rng);
    std::swap(idx[i], idx[j]);
  }
  idx.resize(batch_size);
  return idx;
}

} // namespace cpu_mlp
