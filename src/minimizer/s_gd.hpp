#pragma once

#include "../common.hpp"
#include "minimizer_base.hpp"
#include <Eigen/Eigen>
#include <chrono>
#include <random>
#include <cmath>

/**
 * @brief Simple Stochastic Gradient Descent minimizer.
 *
 * Provides a stochastic solver that operates on minibatches supplied via
 * `setData(...)`. Also implements a full-batch `solve(...)` fallback so
 * the class satisfies the `MinimizerBase` interface.
 */
template <typename V, typename M>
class StochasticGradientDescent : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;

public:
  using S_VecFun = std::function<double(const V &, const V &, const V &)>;
  // MODIFICATO: La funzione gradiente ora riceve il vettore dei pesi, gli INDICI del batch e il vettore gradiente in output
  using BatchGradFun = std::function<void(const V &, const std::vector<size_t>&, V &)>;

  StochasticGradientDescent() = default;

  // same minibatch sampler signature as in SLBFGS
  static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);

  // MODIFICATO: accetta Matrici per inputs e targets
  void setData(const M &inputs, const M &targets, const S_VecFun &f,
               const BatchGradFun &g) {
    // We assume columns are samples
    _inputs = inputs;
    _targets = targets;
    _sf = f;
    _batch_g = g;
  }

  void setStepSize(double s) { this->step_size = s; }

  // Full-batch solve (fallback) — reuse simple GD behavior
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    const bool timing = (this->recorder_ != nullptr);
    if (this->recorder_) this->recorder_->reset();
    auto start_time = std::chrono::steady_clock::now();

    for (_iters = 0; _iters < _max_iters; ++_iters) {
      V g = Gradient(x);
      if (g.norm() < _tol)
        break;
      x = x - this->step_size * g;

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

  // Stochastic solver using minibatches previously set with setData
  V stochastic_solve(const V& init_w, int m /*minibatches per epoch*/, int b /*batch size*/, double step, bool verbose = false,
                     int print_every = 50) {
    if (_inputs.cols() == 0 || _targets.cols() == 0) {
      throw std::runtime_error("No data set for StochasticGradientDescent::stochastic_solve");
    }

    int N = static_cast<int>(_inputs.cols()); // Assume col-major data
    // int dim = static_cast<int>(_inputs[0].size()); // INCORRECT INFERENCE

    V w = init_w;
    int dim = static_cast<int>(w.size());

    double passes = 0.0;
    std::mt19937 rng(123);

    const bool timing = (this->recorder_ != nullptr);
    if (this->recorder_) this->recorder_->reset();
    auto start_time = std::chrono::steady_clock::now();

    while (_iters < _max_iters) {
      // one epoch of m minibatches
      double last_grad_norm = 0.0;
      for (int t = 0; t < m; ++t) {
        // sample minibatch indices using shared helper
        auto minibatch_indices = sample_minibatch_indices(N, b, rng);

        V grad_est = V::Zero(dim);
        
        // MODIFICATO: Chiamata unica batch-wise
        // The callback logic handles extracting data from the pre-set Matrix using indices
        _batch_g(w, minibatch_indices, grad_est);
        last_grad_norm = grad_est.norm();

        // update
        w = w - step * grad_est;

        passes += static_cast<double>(b) / static_cast<double>(N);
      }

      // compute mean loss over dataset and log
      // Note: Computing full loss every epoch is expensive!
      // For performance, we might want to skip this or estimate it.
      // But keeping legacy behavior for now.
      double loss = 0.0;
      // NOTE: Using _sf here which is single-item based is slow.
      // Ideally we would have a BatchLossFun too.
      // But since user asked not to touch comments and minimize changes, we keep this if possible.
      // However, iterating cols of Matrix is easy.
      for (int i = 0; i < N; ++i) {
        loss += _sf(w, _inputs.col(i), _targets.col(i));
      }
      loss /= static_cast<double>(N);

      if (verbose && ((_iters % print_every) == 0)) {
        std::cout << "Epoch " << (_iters + 1) << " loss=" << loss << " passes=" << passes << std::endl;
      }

      if (this->recorder_) {
        double elapsed_ms = 0.0;
        if (timing) {
          auto now = std::chrono::steady_clock::now();
          elapsed_ms =
              std::chrono::duration<double, std::milli>(now - start_time).count();
        }
        this->recorder_->record(_iters, loss, last_grad_norm, elapsed_ms);
      }

      _iters++;
    }

    return w;
  }

private:
  M _inputs;  // Changed from std::vector<V> to Matrix
  M _targets; // Changed from std::vector<V> to Matrix
  S_VecFun _sf;
  BatchGradFun _batch_g; // MODIFICATO: Tipo batch
};

// Implementation of minibatch sampler (same logic as SLBFGS)
template <typename V, typename M>
std::vector<size_t> StochasticGradientDescent<V,M>::sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng){
    if (N == 0 || batch_size == 0) {
        std::cerr << "Warning: sampling minibatch from empty dataset or with batch_size=0." << std::endl;
      return {};
    }

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) {
      idx[i] = i;
    }

    if (batch_size >= N) {
        std::cerr << "Warning: batch_size >= dataset size. Returning full dataset indices." << std::endl;
        return idx;
    }

    // Fisher-Yates partial shuffle
    for (size_t i = 0; i < batch_size; ++i) {
      std::uniform_int_distribution<size_t> dist(i, N -1);
      size_t j = dist(rng);
      std::swap(idx[i], idx[j]);
    }
    idx.resize(batch_size);
    return idx;
}
