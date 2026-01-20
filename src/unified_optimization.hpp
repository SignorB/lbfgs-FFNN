#pragma once

#include "network_wrapper.hpp"
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <type_traits>

// CPU Headers
#include "gd.hpp"
#include "s_lbfgs.hpp"
#include "lbfgs.hpp"
#include "s_gd.hpp"

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


// -----------------------------------------------------------
// Abstract Strategy Wrapper
// -----------------------------------------------------------

/**
 * @class UnifiedOptimizer
 * @brief Abstract base class for backend-specific optimizer strategies.
 * @tparam Backend The computing backend (CpuBackend or CudaBackend).
 */
template <typename Backend>
class UnifiedOptimizer;


// =================================================================================================
//                                         CPU BACKEND
// =================================================================================================

/**
 * @brief Specialization for CPU Backend.
 */
template <>
class UnifiedOptimizer<CpuBackend> {
public:
    virtual ~UnifiedOptimizer() = default;
    
    /**
     * @brief Executes the optimization strategy.
     * @param net The network wrapper.
     * @param data The dataset.
     * @param config Configuration parameters.
     */
    virtual void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) = 0;
};

/**
 * @class UnifiedGD_CPU
 * @brief Standard Gradient Descent implementation for CPU.
 */
class UnifiedGD_CPU : public UnifiedOptimizer<CpuBackend> {
public:
    void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) override {
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        
        auto minimizer = std::make_shared<GradientDescent<Vec, Mat>>();
        minimizer->setMaxIterations(config.max_iters);
        minimizer->setTolerance(config.tolerance);
        minimizer->setStepSize(config.learning_rate);
        minimizer->useLineSearch(false);
        
        net.getInternal().train(data.train_x, data.train_y, minimizer);
    }
};

/**
 * @class UnifiedLBFGS_CPU
 * @brief L-BFGS implementation for CPU.
 */
class UnifiedLBFGS_CPU : public UnifiedOptimizer<CpuBackend> {
public:
    void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) override {
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        
        auto minimizer = std::make_shared<LBFGS<Vec, Mat>>();
        minimizer->setMaxIterations(config.max_iters);
        minimizer->setTolerance(config.tolerance);
        minimizer->setHistorySize(config.m_param > 0 ? config.m_param : 10);
        
        net.getInternal().train(data.train_x, data.train_y, minimizer);
    }
};

/**
 * @class UnifiedSGD_CPU
 * @brief Stochastic Gradient Descent implementation for CPU (Optimized with Batch Matrix Ops).
 */
class UnifiedSGD_CPU : public UnifiedOptimizer<CpuBackend> {
public:
    void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) override {
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        
        auto minimizer = std::make_shared<StochasticGradientDescent<Vec, Mat>>();
        minimizer->setMaxIterations(config.max_iters);
        minimizer->setStepSize(config.learning_rate);
        minimizer->setLogFile("sgd_log.csv"); // Optional logging
        
        std::cout << "Starting Batch SGD (CPU Optimized)..." << std::endl;
        
        int m = static_cast<int>(data.train_x.cols()) / config.batch_size;
        if(m == 0) m = 1;

        // Delegate to efficient Network implementation
        net.getInternal().train_sgd(
            data.train_x, 
            data.train_y, 
            minimizer, 
            m, 
            config.batch_size, 
            config.learning_rate, 
            true, 
            config.log_interval
        );
    }
};

// /**
//  * @class UnifiedSGD_CPU
//  * @brief Stochastic Gradient Descent implementation for CPU.
//  */
// class UnifiedSGD_CPU : public UnifiedOptimizer<CpuBackend> {
// public:
//     void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) override {
//         using Vec = Eigen::VectorXd;
//         using Mat = Eigen::MatrixXd;
        
//         auto minimizer = std::make_shared<StochasticGradientDescent<Vec, Mat>>();
//         minimizer->setMaxIterations(config.max_iters);
//         minimizer->setStepSize(config.learning_rate);
        
//         std::cout << "preparing data..." << std::endl;
//         std::vector<Vec> inputs(data.train_x.cols());
//         std::vector<Vec> targets(data.train_y.cols());
//         for (int i = 0; i < data.train_x.cols(); ++i) {
//             inputs[i] = data.train_x.col(i);
//             targets[i] = data.train_y.col(i);
//         }

//         // Prepare callbacks to wrap Network
//         auto& network = net.getInternal();
        
//         // Loss Function Callback
//         auto f = [&](const Vec &w, const Vec &x, const Vec &y) -> double {
//              network.setParams(w);
             
//              // Forward expects Matrix (n, 1) to match column vector
//              Eigen::MatrixXd input_mat(x.size(), 1);
//              input_mat.col(0) = x;
             
//              const auto &output = network.forward(input_mat);
//              Vec out_vec = output.col(0);

             
//              double loss =  0.5 * (out_vec - y).squaredNorm();
//              return loss;
//         };

//         // Gradient Function Callback
//         auto g = [&](const Vec &w, const Vec &x, const Vec &y, Vec &grad) {
//              network.setParams(w);
//              network.zeroGrads();
             
//              Eigen::MatrixXd input_mat(x.size(), 1);
//              input_mat.col(0) = x;
             
//              const auto &output = network.forward(input_mat);
             
//              // Loss Gradient (assuming MSE loss gradient at output: out - target)
//              Eigen::MatrixXd loss_grad_mat(y.size(), 1);
//              loss_grad_mat.col(0) = output.col(0) - y;
             
//              network.backward(loss_grad_mat); 
             
//              network.getGrads(grad);
//         };

//         std::cout << "setting data..." << std::endl;
//         minimizer->setData(inputs, targets, f, g);
        
//         int m = data.train_x.cols() / config.batch_size;
//         if(m==0) m=1;

//         std::cout << "extracting initial weights..." << std::endl;
//         // Extract initial weights
//         Vec w_init(network.getSize());
//         std::copy(network.getParamsData(), network.getParamsData() + network.getSize(), w_init.data());

//         std::cout << "solving..." << std::endl;
        
//         Vec w_final = minimizer->stochastic_solve(
//             w_init,
//             m, 
//             config.batch_size, 
//             config.learning_rate, 
//             true, 
//             config.log_interval
//         );
        
        
//         network.setParams(w_final);
//     }
// };


/**
 * @class UnifiedSLBFGS_CPU
 * @brief Stochastic L-BFGS implementation for CPU.
 */
class UnifiedSLBFGS_CPU : public UnifiedOptimizer<CpuBackend> {
public:
    void optimize(NetworkWrapper<CpuBackend>& net, const UnifiedDataset& data, const UnifiedConfig& config) override {
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        
        auto minimizer = std::make_shared<SLBFGS<Vec, Mat>>();
        minimizer->setMaxIterations(config.max_iters);
        minimizer->setTolerance(config.tolerance);
        
        int b_H = config.b_H_param > 0 ? config.b_H_param : config.batch_size/2;
        int m = data.train_x.cols() / config.batch_size;
        if(m==0) m=1;

        // No need to set params here, train_stochastic handles it via direct solve call
        
        net.getInternal().train_stochastic(
            data.train_x, 
            data.train_y, 
            minimizer,
            m, 
            config.m_param, 
            config.L_param, 
            config.batch_size, 
            b_H, 
            config.learning_rate, 
            true, 
            config.log_interval
        );
    }
};






#ifdef __CUDACC__
#include "cuda/gd.cuh"
#include "cuda/lbfgs.cuh"
#include "cuda/sgd.cuh"
#include "cuda/iteration_recorder.cuh"
#include "cuda/device_buffer.cuh"

/**
 * @brief Specialization for CUDA Backend.
 */
template <>
class UnifiedOptimizer<CudaBackend> {
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
    virtual void optimize(
        cuda_mlp::CublasHandle& handle,
        NetworkWrapper<CudaBackend>& net, 
        const UnifiedDataset& host_data,
        cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& d_train_x,
        cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& d_train_y,
        const UnifiedConfig& config
    ) = 0;
};

/**
 * @brief Helper to execute CUDA solver loop.
 */
inline void run_cuda_solver(
    std::unique_ptr<cuda_mlp::CudaMinimizerBase> solver,
    cuda_mlp::CudaNetwork& net,
    cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& d_train_x,
    cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& d_train_y,
    const UnifiedDataset& dataset,
    const UnifiedConfig& config
) {
    using namespace cuda_mlp;
    IterationRecorder recorder;
    recorder.init(config.max_iters);
    solver->setRecorder(&recorder);
    
    auto loss_grad = [&](const CudaScalar *params, CudaScalar *grad, const CudaScalar *input, const CudaScalar *target, int batch) -> CudaScalar {
        CudaScalar loss = net.compute_loss_and_grad(input, target, batch);
        device_copy(grad, net.grads_data(), net.params_size());
        return loss;
    };

    solver->solve(net.params_size(),
        net.params_data(),
        d_train_x.data(),
        d_train_y.data(),
        static_cast<int>(dataset.train_x.cols()),
        loss_grad);
        
    cudaDeviceSynchronize();
}

/**
 * @class UnifiedGD_CUDA
 * @brief Gradient Descent for CUDA.
 */
class UnifiedGD_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
    void optimize(cuda_mlp::CublasHandle& handle, NetworkWrapper<CudaBackend>& net, const UnifiedDataset& d, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dx, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dy, const UnifiedConfig& c) override {
        using namespace cuda_mlp;
        auto solver = std::make_unique<CudaGD>(handle);
        solver->setLearningRate(c.learning_rate);
        solver->setMomentum(c.momentum);
        solver->setMaxIterations(c.max_iters);
        solver->setTolerance(c.tolerance);
        run_cuda_solver(std::move(solver), net.getInternal(), dx, dy, d, c);
    }
};

/**
 * @class UnifiedLBFGS_CUDA
 * @brief L-BFGS for CUDA.
 */
class UnifiedLBFGS_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
     void optimize(cuda_mlp::CublasHandle& handle, NetworkWrapper<CudaBackend>& net, const UnifiedDataset& d, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dx, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dy, const UnifiedConfig& c) override {
        using namespace cuda_mlp;
        auto solver = std::make_unique<CudaLBFGS>(handle);
        solver->setMemory(c.m_param); 
        solver->setMaxIterations(c.max_iters);
        solver->setTolerance(c.tolerance);
        run_cuda_solver(std::move(solver), net.getInternal(), dx, dy, d, c);
     }
};

/**
 * @class UnifiedSGD_CUDA
 * @brief Stochastic Gradient Descent for CUDA.
 */
class UnifiedSGD_CUDA : public UnifiedOptimizer<CudaBackend> {
public:
     void optimize(cuda_mlp::CublasHandle& handle, NetworkWrapper<CudaBackend>& net, const UnifiedDataset& d, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dx, cuda_mlp::DeviceBuffer<cuda_mlp::CudaScalar>& dy, const UnifiedConfig& c) override {
        using namespace cuda_mlp;
        auto solver = std::make_unique<CudaSGD>(handle);
        solver->setLearningRate(c.learning_rate);
        solver->setMomentum(c.momentum);
        solver->setBatchSize(c.batch_size);
        solver->setMaxIterations(c.max_iters);
        solver->setDimensions(static_cast<int>(d.train_x.rows()), static_cast<int>(d.train_y.rows()));
        run_cuda_solver(std::move(solver), net.getInternal(), dx, dy, d, c);
     }
};

/**
 * @class UnavailableOptimizer
 * @brief Triggering static assertion for unavailable backend+optimizer combinations.
 */
template <typename T>
class UnavailableOptimizer {
    static_assert(sizeof(T) == 0, "This Optimizer is NOT available on the current Backend (e.g. SLBFGS is CPU-only).");
};

#endif





/**
 * @brief Unified alias for Gradient Descent (CPU & CUDA).
 */
template <typename Backend> 
using UnifiedGD = typename std::conditional<
    std::is_same<Backend, CpuBackend>::value, 
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
using UnifiedLBFGS = typename std::conditional<
    std::is_same<Backend, CpuBackend>::value, 
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
using UnifiedSGD = typename std::conditional<
    std::is_same<Backend, CpuBackend>::value, 
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
using UnifiedSLBFGS = typename std::conditional<
    std::is_same<Backend, CpuBackend>::value, 
    UnifiedSLBFGS_CPU,
    #ifdef __CUDACC__
    UnavailableOptimizer<Backend> 
    #else
    void
    #endif
>::type;
