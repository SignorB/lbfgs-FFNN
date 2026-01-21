#pragma once

#include "iteration_recorder.hpp"
#include "network_wrapper.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// CPU Headers
#include "minimizer/gd.hpp"
#include "minimizer/lbfgs.hpp"
#include "minimizer/s_gd.hpp"
#include "minimizer/s_lbfgs.hpp"

/**
 * @struct UnifiedConfig
 * @brief Configuration parameters for training experiments.
 */
struct UnifiedConfig {
  std::string name = "Experiment";

  int max_iters = 100;
  double tolerance = 1e-4;
  double learning_rate = 0.01;
  double momentum = 0.0;

  // Stochastic params
  int batch_size = 128;
  int m_param = 10;
  int L_param = 10;
  int b_H_param = 0;

  // Logging
  int log_interval = 10;
};

/**
 * @struct UnifiedDataset
 * @brief Container for training and test data.
 */
struct UnifiedDataset {
  Eigen::MatrixXd train_x;
  Eigen::MatrixXd train_y;
  Eigen::MatrixXd test_x;
  Eigen::MatrixXd test_y;
};

inline std::string cpu_log_filename(const UnifiedConfig &config) {
  std::string base = config.name.empty() ? "run" : config.name;
  return base + "_history.csv";
}

inline void write_cpu_history_csv(
    const std::string &filename, const IterationRecorder<CpuBackend> &recorder, int log_interval) {
  if (log_interval <= 0) return;
  std::vector<double> loss_hist;
  std::vector<double> grad_hist;
  std::vector<double> time_hist;
  recorder.copy_to_host(loss_hist, grad_hist, time_hist);
  if (loss_hist.empty()) return;

  std::ofstream log_file(filename);
  if (!log_file.is_open()) return;
  log_file << "Iteration,Loss,GradNorm,TimeMs\n";
  int stride = std::max(1, log_interval);
  for (size_t i = 0; i < loss_hist.size(); i += static_cast<size_t>(stride)) {
    double loss = loss_hist[i];
    double grad = (i < grad_hist.size()) ? grad_hist[i] : 0.0;
    double time_ms = (i < time_hist.size()) ? time_hist[i] : 0.0;
    log_file << i << "," << loss << "," << grad << "," << time_ms << "\n";
  }
}

inline void run_full_batch_cpu(NetworkWrapper<CpuBackend> &net,
    const UnifiedDataset &data,
    cpu_mlp::FullBatchMinimizer<Eigen::VectorXd, Eigen::MatrixXd> &minimizer) {
  using Vec = Eigen::VectorXd;
  using Mat = Eigen::MatrixXd;

  auto &network = net.getInternal();
  size_t params_size = network.getSize();

  Vec weights(params_size);
  std::copy(network.getParamsData(), network.getParamsData() + params_size, weights.data());

  const double inv_samples = (data.train_x.cols() > 0) ? (1.0 / static_cast<double>(data.train_x.cols())) : 0.0;

  VecFun<Vec, double> f = [&](Vec w) -> double {
    network.setParams(w);
    const auto &output = network.forward(data.train_x);
    Mat diff = output - data.train_y;
    double loss = 0.5 * diff.squaredNorm();
    if (inv_samples != 0.0) loss *= inv_samples;
    return loss;
  };

  GradFun<Vec> grad = [&](Vec w) -> Vec {
    network.setParams(w);
    network.zeroGrads();
    const auto &output = network.forward(data.train_x);
    Mat diff = output - data.train_y;
    network.backward(diff);
    Vec g(params_size);
    network.getGrads(g);
    if (inv_samples != 0.0) g *= inv_samples;
    return g;
  };

  Vec final_weights = minimizer.solve(weights, f, grad);
  network.setParams(final_weights);
}

// -----------------------------------------------------------
// Abstract Strategy Wrapper
// -----------------------------------------------------------

/**
 * @class UnifiedOptimizer
 * @brief Abstract base class for backend-specific optimizer strategies.
 * @tparam Backend The computing backend (CpuBackend or CudaBackend).
 */
template <typename Backend> class UnifiedOptimizer;

// =================================================================================================
//                                         CPU BACKEND
// =================================================================================================

/**
 * @brief Specialization for CPU Backend.
 */
template <> class UnifiedOptimizer<CpuBackend> {
public:
  virtual ~UnifiedOptimizer() = default;

  /**
   * @brief Executes the optimization strategy.
   * @param net The network wrapper.
   * @param data The dataset.
   * @param config Configuration parameters.
   */
  virtual void optimize(NetworkWrapper<CpuBackend> &net, const UnifiedDataset &data, const UnifiedConfig &config) = 0;
};

/**
 * @class UnifiedGD_CPU
 * @brief Standard Gradient Descent implementation for CPU.
 */
class UnifiedGD_CPU : public UnifiedOptimizer<CpuBackend> {
public:
  void optimize(NetworkWrapper<CpuBackend> &net, const UnifiedDataset &data, const UnifiedConfig &config) override {
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    auto minimizer = std::make_shared<cpu_mlp::GradientDescent<Vec, Mat>>();
    minimizer->setMaxIterations(config.max_iters);
    minimizer->setTolerance(config.tolerance);
    minimizer->setStepSize(config.learning_rate);
    minimizer->useLineSearch(false);
    IterationRecorder<CpuBackend> recorder;
    recorder.init(config.max_iters);
    minimizer->setRecorder(&recorder);

    run_full_batch_cpu(net, data, *minimizer);
    write_cpu_history_csv(cpu_log_filename(config), recorder, config.log_interval);
  }
};

/**
 * @class UnifiedLBFGS_CPU
 * @brief L-BFGS implementation for CPU.
 */
class UnifiedLBFGS_CPU : public UnifiedOptimizer<CpuBackend> {
public:
  void optimize(NetworkWrapper<CpuBackend> &net, const UnifiedDataset &data, const UnifiedConfig &config) override {
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    auto minimizer = std::make_shared<cpu_mlp::LBFGS<Vec, Mat>>();
    minimizer->setMaxIterations(config.max_iters);
    minimizer->setTolerance(config.tolerance);
    minimizer->setHistorySize(config.m_param > 0 ? config.m_param : 10);
    IterationRecorder<CpuBackend> recorder;
    recorder.init(config.max_iters);
    minimizer->setRecorder(&recorder);

    run_full_batch_cpu(net, data, *minimizer);
    write_cpu_history_csv(cpu_log_filename(config), recorder, config.log_interval);
  }
};

/**
 * @class UnifiedSGD_CPU
 * @brief Stochastic Gradient Descent implementation for CPU (Optimized with Batch Matrix Ops).
 */
class UnifiedSGD_CPU : public UnifiedOptimizer<CpuBackend> {
public:
  void optimize(NetworkWrapper<CpuBackend> &net, const UnifiedDataset &data, const UnifiedConfig &config) override {
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    auto minimizer = std::make_shared<cpu_mlp::StochasticGradientDescent<Vec, Mat>>();
    minimizer->setMaxIterations(config.max_iters);
    minimizer->setStepSize(config.learning_rate);

    IterationRecorder<CpuBackend> recorder;
    recorder.init(config.max_iters);
    minimizer->setRecorder(&recorder);

    std::cout << "Starting Batch SGD (CPU Optimized)..." << std::endl;

    int m = static_cast<int>(data.train_x.cols()) / config.batch_size;
    if (m == 0) m = 1;

    auto &network = net.getInternal();
    size_t params_size = network.getSize();

    Vec weights(params_size);
    std::copy(network.getParamsData(), network.getParamsData() + params_size, weights.data());

    long input_rows = data.train_x.rows();
    long output_rows = data.train_y.rows();

    Mat batch_x_buffer(input_rows, config.batch_size);
    Mat batch_y_buffer(output_rows, config.batch_size);

    auto batch_g = [&](const Vec &w, const std::vector<size_t> &indices, Vec &grad) mutable {
      network.setParams(w);
      network.zeroGrads();

      long current_bs = indices.size();

      if (batch_x_buffer.cols() != current_bs) {
        batch_x_buffer.resize(input_rows, current_bs);
        batch_y_buffer.resize(output_rows, current_bs);
      }

      for (long i = 0; i < current_bs; ++i) {
        batch_x_buffer.col(i) = data.train_x.col(indices[i]);
        batch_y_buffer.col(i) = data.train_y.col(indices[i]);
      }

      const auto &output = network.forward(batch_x_buffer);

      Mat diff = output - batch_y_buffer;
      network.backward(diff);

      network.getGrads(grad);
      grad /= static_cast<double>(current_bs);
    };

    auto f_single = [&](const Vec &w, const Vec &x, const Vec &y) -> double {
      network.setParams(w);
      Eigen::MatrixXd input_mat(x.size(), 1);
      input_mat.col(0) = x;
      const auto &output = network.forward(input_mat);
      return 0.5 * (output.col(0) - y).squaredNorm();
    };

    minimizer->setData(data.train_x, data.train_y, f_single, batch_g);

    Vec final_weights = minimizer->stochastic_solve(weights,
        m,
        config.batch_size,
        config.learning_rate,
        true, // verbose
        config.log_interval);
    write_cpu_history_csv(cpu_log_filename(config), recorder, config.log_interval);
  }
};

/**
 * @class UnifiedSLBFGS_CPU
 * @brief Stochastic L-BFGS implementation for CPU.
 */
class UnifiedSLBFGS_CPU : public UnifiedOptimizer<CpuBackend> {
public:
  void optimize(NetworkWrapper<CpuBackend> &net, const UnifiedDataset &data, const UnifiedConfig &config) override {
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    auto minimizer = std::make_shared<cpu_mlp::SLBFGS<Vec, Mat>>();
    minimizer->setMaxIterations(config.max_iters);
    minimizer->setTolerance(config.tolerance);
    IterationRecorder<CpuBackend> recorder;
    recorder.init(config.max_iters);
    minimizer->setRecorder(&recorder);

    int b_H = config.b_H_param > 0 ? config.b_H_param : config.batch_size / 2;
    int m = data.train_x.cols() / config.batch_size;
    if (m == 0) m = 1;

    auto &network = net.getInternal();
    size_t params_size = network.getSize();

    Vec weights(params_size);
    std::copy(network.getParamsData(), network.getParamsData() + params_size, weights.data());
    double lambda = 1e-4;

    long input_rows = data.train_x.rows();
    long output_rows = data.train_y.rows();
    int N = data.train_x.cols();

    Mat batch_x_buffer(input_rows, std::max(config.batch_size, 128));
    Mat batch_y_buffer(output_rows, std::max(config.batch_size, 128));

    auto batch_g = [&](const Vec &w, const std::vector<size_t> &indices, Vec &grad) mutable {
      network.setParams(w);
      network.zeroGrads();

      long current_bs = indices.size();

      if (batch_x_buffer.cols() < current_bs) {
        batch_x_buffer.resize(input_rows, current_bs);
        batch_y_buffer.resize(output_rows, current_bs);
      }

      bool is_full_batch = (current_bs == N);

      if (is_full_batch) {
        const auto &output = network.forward(data.train_x);
        Mat diff = output - data.train_y;
        network.backward(diff);
      } else {
        for (long i = 0; i < current_bs; ++i) {
          batch_x_buffer.col(i) = data.train_x.col(indices[i]);
          batch_y_buffer.col(i) = data.train_y.col(indices[i]);
        }
        auto x_view = batch_x_buffer.leftCols(current_bs);
        auto y_view = batch_y_buffer.leftCols(current_bs);

        const auto &output = network.forward(x_view);
        Mat diff = output - y_view;
        network.backward(diff);
      }

      network.getGrads(grad);
      grad /= static_cast<double>(current_bs);
      grad.array() += lambda * w.array();
    };

    auto batch_f = [&](const Vec &w, const std::vector<size_t> &indices) -> double {
      network.setParams(w);
      long current_bs = indices.size();

      if (batch_x_buffer.cols() < current_bs) {
        batch_x_buffer.resize(input_rows, current_bs);
        batch_y_buffer.resize(output_rows, current_bs);
      }

      for (long i = 0; i < current_bs; ++i) {
        batch_x_buffer.col(i) = data.train_x.col(indices[i]);
        batch_y_buffer.col(i) = data.train_y.col(indices[i]);
      }
      auto x_view = batch_x_buffer.leftCols(current_bs);
      auto y_view = batch_y_buffer.leftCols(current_bs);

      const auto &output = network.forward(x_view);
      Vec diff_sq = (output - y_view).colwise().squaredNorm();
      double loss = 0.5 * diff_sq.sum();
      loss /= current_bs;
      loss += 0.5 * lambda * w.squaredNorm();
      return loss;
    };

    minimizer->setData(batch_f, batch_g);

    Vec final_weights = minimizer->stochastic_solve(
        weights, batch_f, batch_g, m, config.m_param, config.L_param, config.batch_size, b_H, config.learning_rate, N);
    write_cpu_history_csv(cpu_log_filename(config), recorder, config.log_interval);
  }
};

#ifdef __CUDACC__
  #include "cuda/device_buffer.cuh"
  #include "cuda/gd.cuh"
  #include "cuda/lbfgs.cuh"
  #include "cuda/sgd.cuh"
  #include "iteration_recorder.hpp"

/**
 * @brief Specialization for CUDA Backend.
 */
template <> class UnifiedOptimizer<CudaBackend> {
public:
  virtual ~UnifiedOptimizer() = default;

  /**
   * @brief Executes the optimization strategy on GPU.
   * @param handle Allocator/Cublas handle.
   * @param net Network wrapper.
   * @param host_data Original host dataset (for dimensions).
   * @param d_train_x Device buffer for input.
   * @param d_train_y Device buffer for targets.
   * @param config Configuration parameters.
   */
  virtual void optimize(cuda_mlp::CublasHandle &handle,
      NetworkWrapper<CudaBackend> &net,
      const UnifiedDataset &host_data,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &d_train_x,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &d_train_y,
      const UnifiedConfig &config) = 0;
};

inline std::string cuda_log_filename(const UnifiedConfig &config) {
  std::string base = config.name.empty() ? "run" : config.name;
  return base + "_history.csv";
}

inline void write_cuda_history_csv(
    const std::string &filename, const IterationRecorder<CudaBackend> &recorder, int log_interval) {
  if (log_interval <= 0) return;
  std::vector<cuda_mlp::CudaScalar> loss_hist;
  std::vector<cuda_mlp::CudaScalar> grad_hist;
  std::vector<cuda_mlp::CudaScalar> time_hist;
  recorder.copy_to_host(loss_hist, grad_hist, time_hist);
  if (loss_hist.empty()) return;

  std::ofstream log_file(filename);
  if (!log_file.is_open()) return;
  log_file << "Iteration,Loss,GradNorm,TimeMs\n";
  int stride = std::max(1, log_interval);
  for (size_t i = 0; i < loss_hist.size(); i += static_cast<size_t>(stride)) {
    cuda_mlp::CudaScalar loss = loss_hist[i];
    cuda_mlp::CudaScalar grad = (i < grad_hist.size()) ? grad_hist[i] : static_cast<cuda_mlp::CudaScalar>(0);
    cuda_mlp::CudaScalar time_ms = (i < time_hist.size()) ? time_hist[i] : static_cast<cuda_mlp::CudaScalar>(0);
    log_file << i << "," << loss << "," << grad << "," << time_ms << "\n";
  }
}

/**
 * @brief Helper to execute CUDA solver loop with multiple runs and logging.
 */
template <typename SolverFactory>
inline void run_cuda_solver_once(SolverFactory make_solver,
    cuda_mlp::CudaNetwork &net,
    cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &d_train_x,
    cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &d_train_y,
    const UnifiedDataset &dataset,
    const UnifiedConfig &config) {
  using namespace cuda_mlp;
  DeviceBuffer<CudaScalar> initial_params(net.params_size());
  if (net.params_size() > 0) {
    device_copy(initial_params.data(), net.params_data(), net.params_size());
  }

  auto loss_grad = [&](const CudaScalar *params,
                       CudaScalar *grad,
                       const CudaScalar *input,
                       const CudaScalar *target,
                       int batch) -> CudaScalar {
    CudaScalar loss = net.compute_loss_and_grad(input, target, batch);
    device_copy(grad, net.grads_data(), net.params_size());
    return loss;
  };

  if (net.params_size() > 0) {
    device_copy(net.params_data(), initial_params.data(), net.params_size());
  }

  auto solver = make_solver();
  IterationRecorder<CudaBackend> recorder;
  recorder.init(config.max_iters);
  solver->setRecorder(&recorder);

  auto start_time = std::chrono::steady_clock::now();
  solver->solve(net.params_size(),
      net.params_data(),
      d_train_x.data(),
      d_train_y.data(),
      static_cast<int>(dataset.train_x.cols()),
      loss_grad);
  cudaDeviceSynchronize();
  auto end_time = std::chrono::steady_clock::now();
  (void)start_time;
  (void)end_time;

  write_cuda_history_csv(cuda_log_filename(config), recorder, config.log_interval);
}

/**
 * @class UnifiedGD_CUDA
 * @brief Gradient Descent for CUDA.
 */
class UnifiedGD_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
  void optimize(cuda_mlp::CublasHandle &handle,
      NetworkWrapper<CudaBackend> &net,
      const UnifiedDataset &d,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dx,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dy,
      const UnifiedConfig &c) override {
    using namespace cuda_mlp;
    run_cuda_solver_once(
        [&]() {
          auto solver = std::make_unique<CudaGD>(handle);
          solver->setLearningRate(c.learning_rate);
          solver->setMomentum(c.momentum);
          solver->setMaxIterations(c.max_iters);
          solver->setTolerance(c.tolerance);
          return solver;
        },
        net.getInternal(),
        dx,
        dy,
        d,
        c);
  }
};

/**
 * @class UnifiedLBFGS_CUDA
 * @brief L-BFGS for CUDA.
 */
class UnifiedLBFGS_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
  void optimize(cuda_mlp::CublasHandle &handle,
      NetworkWrapper<CudaBackend> &net,
      const UnifiedDataset &d,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dx,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dy,
      const UnifiedConfig &c) override {
    using namespace cuda_mlp;
    run_cuda_solver_once(
        [&]() {
          auto solver = std::make_unique<CudaLBFGS>(handle);
          solver->setMemory(c.m_param);
          solver->setMaxIterations(c.max_iters);
          solver->setTolerance(c.tolerance);
          return solver;
        },
        net.getInternal(),
        dx,
        dy,
        d,
        c);
  }
};

/**
 * @class UnifiedSGD_CUDA
 * @brief Stochastic Gradient Descent for CUDA.
 */
class UnifiedSGD_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
  void optimize(cuda_mlp::CublasHandle &handle,
      NetworkWrapper<CudaBackend> &net,
      const UnifiedDataset &d,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dx,
      cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar> &dy,
      const UnifiedConfig &c) override {
    using namespace cuda_mlp;
    run_cuda_solver_once(
        [&]() {
          auto solver = std::make_unique<CudaSGD>(handle);
          solver->setLearningRate(c.learning_rate);
          solver->setMomentum(c.momentum);
          solver->setBatchSize(c.batch_size);
          solver->setMaxIterations(c.max_iters);
          solver->setDimensions(static_cast<int>(d.train_x.rows()), static_cast<int>(d.train_y.rows()));
          return solver;
        },
        net.getInternal(),
        dx,
        dy,
        d,
        c);
  }
};

/**
 * @class UnavailableOptimizer
 * @brief Triggering static assertion for unavailable backend+optimizer combinations.
 */
template <typename T> class UnavailableOptimizer {
  static_assert(sizeof(T) == 0, "This Optimizer is NOT available on the current Backend (e.g. SLBFGS is CPU-only).");
};

#endif

/**
 * @brief Unified alias for Gradient Descent (CPU & CUDA).
 */
template <typename Backend>
using UnifiedGD = typename std::conditional<std::is_same<Backend, CpuBackend>::value,
    UnifiedGD_CPU,
#ifdef __CUDACC__
    UnifiedGD_CUDA
#else
    void
#endif
    >::type;

/**
 * @brief Unified alias for L-BFGS (CPU & CUDA).
 */
template <typename Backend>
using UnifiedLBFGS = typename std::conditional<std::is_same<Backend, CpuBackend>::value,
    UnifiedLBFGS_CPU,
#ifdef __CUDACC__
    UnifiedLBFGS_CUDA
#else
    void
#endif
    >::type;

/**
 * @brief Unified alias for Stochastic Gradient Descent (CPU & CUDA).
 */
template <typename Backend>
using UnifiedSGD = typename std::conditional<std::is_same<Backend, CpuBackend>::value,
    UnifiedSGD_CPU,
#ifdef __CUDACC__
    UnifiedSGD_CUDA
#else
    void
#endif
    >::type;

/**
 * @brief Unified alias for Stochastic L-BFGS (CPU ONLY).
 * Triggers compile-time error if used with CudaBackend.
 */
template <typename Backend>
using UnifiedSLBFGS = typename std::conditional<std::is_same<Backend, CpuBackend>::value,
    UnifiedSLBFGS_CPU,
#ifdef __CUDACC__
    UnavailableOptimizer<Backend>
#else
    void
#endif
    >::type;
