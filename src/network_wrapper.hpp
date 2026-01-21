#pragma once

/**
 * @file network_wrapper.hpp
 * @brief Backend-agnostic wrapper for CPU/CUDA networks.
 */

#include "layer.hpp"   // CPU layer tags
#include "network.hpp" // CPU implementation

#ifdef __CUDACC__
  #include "cuda/kernels.cuh" // CUDA Enums
  #include "cuda/network.cuh" // GPU Implementation
#endif

#include <iostream>
#include <memory>

/// @brief Backend tag for CPU implementations.
struct CpuBackend {};
/// @brief Backend tag for CUDA implementations.
struct CudaBackend {};

/**
 * @brief Map CPU activation tags to CUDA activation enums.
 * @tparam T Activation tag type.
 */
template <typename T> struct ActivationToEnum;

/// @brief CPU->CUDA activation mapping for Linear.
template <> struct ActivationToEnum<cpu_mlp::Linear> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Linear;
#endif
};
/// @brief CPU->CUDA activation mapping for Sigmoid.
template <> struct ActivationToEnum<cpu_mlp::Sigmoid> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Sigmoid;
#endif
};
/// @brief CPU->CUDA activation mapping for Tanh.
template <> struct ActivationToEnum<cpu_mlp::Tanh> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Tanh;
#endif
};
/// @brief CPU->CUDA activation mapping for ReLU.
template <> struct ActivationToEnum<cpu_mlp::ReLU> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::ReLU;
#endif
};

template <typename Backend> class NetworkWrapper;

/**
 * @brief CPU specialization of the network wrapper.
 */
template <> class NetworkWrapper<CpuBackend> {
public:
  using InternalNetwork = cpu_mlp::Network;

  NetworkWrapper() = default;

  template <int In, int Out, typename Activation> void addLayer() { network_.addLayer<In, Out, Activation>(); }

  void bindParams() { network_.bindParams(); }
  void bindParams(unsigned int seed) { network_.bindParams(seed); }

  /// @brief Access the underlying CPU network.
  InternalNetwork &getInternal() { return network_; }
  /// @brief Access the underlying CPU network (const).
  const InternalNetwork &getInternal() const { return network_; }

  /// @brief Total number of parameters.
  size_t getParamsSize() const { return network_.getSize(); }

private:
  InternalNetwork network_;
};

/**
 * @brief CUDA specialization of the network wrapper.
 */
#ifdef __CUDACC__
template <> class NetworkWrapper<CudaBackend> {
public:
  using InternalNetwork = cuda_mlp::CudaNetwork;

  explicit NetworkWrapper(cuda_mlp::CublasHandle &handle) : network_(handle) {}

  template <int In, int Out, typename Activation> void addLayer() {
    network_.addLayer(In, Out, ActivationToEnum<Activation>::value);
  }

  void bindParams() { network_.bindParams(); }
  void bindParams(unsigned int seed) { network_.bindParams(seed); }

  /// @brief Access the underlying CUDA network.
  InternalNetwork &getInternal() { return network_; }
  /// @brief Access the underlying CUDA network (const).
  const InternalNetwork &getInternal() const { return network_; }

  /// @brief Total number of parameters.
  size_t getParamsSize() const { return network_.params_size(); }

private:
  InternalNetwork network_;
};
#endif
