#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"
#include <Eigen/Eigen>
#include <random>
#include <fstream>
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

  // MODIFICATO: accetta BatchGradFun invece di S_GradFun
  void setData(const std::vector<V> &inputs, const std::vector<V> &targets, const S_VecFun &f,
               const BatchGradFun &g) {
    _inputs = inputs;
    _targets = targets;
    _sf = f;
    _batch_g = g;
  }

  void setLogFile(const std::string &path) { _logfile = path; }
  void setStepSize(double s) { this->step_size = s; }

  // Full-batch solve (fallback) — reuse simple GD behavior
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    for (_iters = 0; _iters < _max_iters; ++_iters) {
      V g = Gradient(x);
      if (g.norm() < _tol)
        break;
      x = x - this->step_size * g;
    }
    return x;
  }

  // Stochastic solver using minibatches previously set with setData
  V stochastic_solve(const V& init_w, int m /*minibatches per epoch*/, int b /*batch size*/, double step, bool verbose = false,
                     int print_every = 50) {
    if (_inputs.empty() || _targets.empty()) {
      throw std::runtime_error("No data set for StochasticGradientDescent::stochastic_solve");
    }

    int N = static_cast<int>(_inputs.size());
    // int dim = static_cast<int>(_inputs[0].size()); // INCORRECT INFERENCE

    V w = init_w;
    int dim = static_cast<int>(w.size());

    // prepare logfile (truncate each run)
    std::ofstream logfile;
    if (!_logfile.empty()) {
      logfile.open(_logfile, std::ofstream::out | std::ofstream::trunc);
      if (logfile.is_open()) {
        logfile << "passes,loss,log10_loss,iteration" << std::endl;
        logfile.flush();
      }
    }

    double passes = 0.0;
    std::mt19937 rng(123);

    while (_iters < _max_iters) {
      // one epoch of m minibatches
      for (int t = 0; t < m; ++t) {
        // sample minibatch indices using shared helper
        auto minibatch_indices = sample_minibatch_indices(N, b, rng);

        V grad_est = V::Zero(dim);
        
        // MODIFICATO: Chiamata unica batch-wise
        _batch_g(w, minibatch_indices, grad_est);

        // update
        w = w - step * grad_est;

        passes += static_cast<double>(b) / static_cast<double>(N);
      }

      // compute mean loss over dataset and log
      double loss = 0.0;
      for (int i = 0; i < N; ++i) {
        loss += _sf(w, _inputs[static_cast<size_t>(i)], _targets[static_cast<size_t>(i)]);
      }
//      mean_loss /= static_cast<double>(N);

      if (logfile.is_open()) {
        double logv = (loss > 0.0) ? std::log10(loss) : -INFINITY;
        logfile << passes << "," << loss << "," << logv << "," << (_iters + 1) << std::endl;
        logfile.flush();
      }

      if (verbose && ((_iters % print_every) == 0)) {
        std::cout << "Epoch " << (_iters + 1) << " loss=" << loss << " passes=" << passes << std::endl;
      }

      _iters++;
    }

    if (logfile.is_open()) logfile.close();
    return w;
  }

private:
  std::vector<V> _inputs;
  std::vector<V> _targets;
  S_VecFun _sf;
  BatchGradFun _batch_g; // MODIFICATO: Tipo batch
  std::string _logfile;
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
