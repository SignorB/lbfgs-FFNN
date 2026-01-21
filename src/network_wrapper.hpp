#pragma once

#include "layer.hpp"   // CPU Layer Types
#include "network.hpp" // CPU Implementation

#ifdef __CUDACC__
  #include "cuda/kernels.cuh" // CUDA Enums
  #include "cuda/network.cuh" // GPU Implementation
#endif

#include <iostream>
#include <memory>

// Backend Tags
struct CpuBackend {};
struct CudaBackend {};

// Trait to map CPU Activation types to CUDA ActivationType enum
template <typename T> struct ActivationToEnum;

// Default/CPU mappings (not used for runtime enum but for checks if needed)
// Default/CPU mappings (not used for runtime enum but for checks if needed)
template <> struct ActivationToEnum<cpu_mlp::Linear> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Linear;
#endif
};
template <> struct ActivationToEnum<cpu_mlp::Sigmoid> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Sigmoid;
#endif
};
template <> struct ActivationToEnum<cpu_mlp::Tanh> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::Tanh;
#endif
};
template <> struct ActivationToEnum<cpu_mlp::ReLU> {
#ifdef __CUDACC__
  static constexpr cuda_mlp::ActivationType value = cuda_mlp::ActivationType::ReLU;
#endif
};

template <typename Backend> class NetworkWrapper;

// --- CPU Specialization ---
template <> class NetworkWrapper<CpuBackend> {
public:
  using InternalNetwork = cpu_mlp::Network;

  NetworkWrapper() = default;

  template <int In, int Out, typename Activation> void addLayer() { network_.addLayer<In, Out, Activation>(); }

  void bindParams() { network_.bindParams(); }

  InternalNetwork &getInternal() { return network_; }
  const InternalNetwork &getInternal() const { return network_; }

  size_t getParamsSize() const { return network_.getSize(); }

private:
  InternalNetwork network_;
};

// --- CUDA Specialization ---
#ifdef __CUDACC__
template <> class NetworkWrapper<CudaBackend> {
public:
  using InternalNetwork = cuda_mlp::CudaNetwork;

  explicit NetworkWrapper(cuda_mlp::CublasHandle &handle) : network_(handle) {}

  template <int In, int Out, typename Activation> void addLayer() {
    network_.addLayer(In, Out, ActivationToEnum<Activation>::value);
  }

  void bindParams() { network_.bindParams(); }

  InternalNetwork &getInternal() { return network_; }
  const InternalNetwork &getInternal() const { return network_; }

  size_t getParamsSize() const { return network_.params_size(); }

private:
  InternalNetwork network_;
};
#endif
