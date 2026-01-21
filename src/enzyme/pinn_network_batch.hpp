#pragma once
/**
 * @file pinn_network_batch.hpp
 * @brief Batched PINN forward utilities with aligned memory.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

/// @brief Scalar type for batched PINN utilities.
using Real = double;

/// @brief Force inline expansion in performance-critical paths.
#ifndef ENZYME_INLINE
  #define ENZYME_INLINE __attribute__((always_inline)) inline
#endif

/// @brief Allocate aligned storage for Real values.
inline Real *aligned_alloc_real(size_t count) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 64, count * sizeof(Real)) != 0) return nullptr;
  return static_cast<Real *>(ptr);
}

/// @brief Free aligned storage allocated by aligned_alloc_real.
inline void aligned_free_real(Real *ptr) { free(ptr); }

/// @brief Tanh activation.
struct Tanh {
  static ENZYME_INLINE Real apply(Real x) { return std::tanh(x); }
};
/// @brief Linear activation.
struct Linear {
  static ENZYME_INLINE Real apply(Real x) { return x; }
};

/**
 * @brief Compile-time dense layer definition with batched forward.
 */
template <int In, int Out, typename Activation> struct Dense {
  static constexpr int InSize = In;
  static constexpr int OutSize = Out;
  static constexpr int NumParams = (In * Out) + Out;

  /// @brief Forward pass for a batch of N inputs.
  static ENZYME_INLINE void forward_batch(
      const Real *__restrict__ p, const Real *__restrict__ in, Real *__restrict__ out, int N) {
    const Real *W = p;
    const Real *b = p + (In * Out);

    for (int k = 0; k < N; ++k) {
      const Real *in_k = in + k * In;
      Real *out_k = out + k * Out;

      for (int i = 0; i < Out; ++i) {
        Real z = 0.0;
        for (int j = 0; j < In; ++j) {
          z += W[i * In + j] * in_k[j];
        }
        z += b[i];
        out_k[i] = Activation::apply(z);
      }
    }
  }
};

/**
 * @brief Compile-time batched network with manually managed parameters.
 */
template <typename... Layers> class PINN {
public:
  using Architecture = std::tuple<Layers...>;
  static constexpr int TotalParams = (0 + ... + Layers::NumParams);
  static constexpr int MaxLayerSize = std::max({Layers::InSize..., Layers::OutSize...});

  /// @brief Raw parameter buffer.
  Real *params;

  PINN() {
    params = aligned_alloc_real(TotalParams + 64);
    if (!params) throw std::runtime_error("Failed to allocate aligned memory for PINN params");
    init_params();
  }

  ~PINN() {
    if (params) aligned_free_real(params);
  }

  /// @brief Disallow copy to avoid double-free of aligned storage.
  PINN(const PINN &) = delete;
  PINN &operator=(const PINN &) = delete;

  /// @brief Initialize parameters with a small uniform range.
  void init_params() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Real> dist(-0.1, 0.1);
    for (int i = 0; i < TotalParams; ++i)
      params[i] = dist(gen);
  }

  /// @brief Forward pass using caller-provided scratch buffers.
  static ENZYME_INLINE void forward_scratch(const Real *in, Real *out, const Real *p, int N, Real *scratch) {
    using L1 = std::tuple_element_t<0, Architecture>;
    using L2 = std::tuple_element_t<1, Architecture>;
    using L3 = std::tuple_element_t<2, Architecture>;
    using L4 = std::tuple_element_t<3, Architecture>;

    Real *bufA = scratch;
    Real *bufB = scratch + (N * MaxLayerSize);

    const Real *p1 = p;
    const Real *p2 = p1 + L1::NumParams;
    const Real *p3 = p2 + L2::NumParams;
    const Real *p4 = p3 + L3::NumParams;

    // Apply each layer in sequence.
    L1::forward_batch(p1, in, bufA, N);
    L2::forward_batch(p2, bufA, bufB, N);
    L3::forward_batch(p3, bufB, bufA, N);
    L4::forward_batch(p4, bufA, out, N);
  }
};
