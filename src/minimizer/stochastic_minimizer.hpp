#pragma once

#include "../common.hpp"
#include "../iteration_recorder.hpp"
#include <Eigen/Eigen>
#include <vector>
#include <random>

namespace cpu_mlp {

/**
 * @brief Base class for Stochastic Minimizers.
 * @details Provides common storage for stochastic optimization parameters like step size and iteration limits.
 */
template <typename V, typename M>
class StochasticMinimizer {
public:
    virtual ~StochasticMinimizer() = default;
    
    /**
     * @brief Sets the maximum number of iterations.
     * @param max_iters Limit on iterations.
     */
    void setMaxIterations(int max_iters) { _max_iters = max_iters; }

    /**
     * @brief Sets the step size (learning rate).
     * @param s Step size.
     */
    void setStepSize(double s) { step_size = s; }

    /**
     * @brief Sets the tolerance for convergence (full gradient norm).
     * @param tol Tolerance value.
     */
    void setTolerance(double tol) { _tol = tol; }
    /**
     * @brief Attach a recorder for loss/grad history.
     * @param recorder Recorder instance (may be null).
     */
    void setRecorder(::IterationRecorder<CpuBackend> *recorder) { recorder_ = recorder; }

protected:
    unsigned int _max_iters = 1000;
    unsigned int _iters = 0;
    double _tol = 1e-4;
    double step_size = 0.01;
    ::IterationRecorder<CpuBackend> *recorder_ = nullptr; ///< Optional recorder for diagnostics
};

} // namespace cpu_mlp
