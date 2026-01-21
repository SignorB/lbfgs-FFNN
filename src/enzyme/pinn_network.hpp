#pragma once
/**
 * @file pinn_network.hpp
 * @brief Minimal feedforward network utilities used for Enzyme experiments.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

/// @brief Scalar type for PINN utilities.
using Real = double;

/// @brief Force inline expansion in performance-critical paths.
#ifndef ENZYME_INLINE
  #define ENZYME_INLINE __attribute__((always_inline)) inline
#endif

/// @brief Tanh activation.
struct Tanh {
  static ENZYME_INLINE Real apply(Real x) { return std::tanh(x); }
};

/// @brief Linear activation.
struct Linear {
  static ENZYME_INLINE Real apply(Real x) { return x; }
};

/**
 * @brief Compile-time dense layer definition.
 */
template <int In, int Out, typename Activation> struct Dense {
  static constexpr int InSize = In;
  static constexpr int OutSize = Out;
  static constexpr int NumParams = (In * Out) + Out;

  /// @brief Forward pass for a single input vector.
  static ENZYME_INLINE void forward(const Real *p, const Real *in, Real *out) {
    const Real *W = p;
    const Real *b = p + (In * Out);

    for (int i = 0; i < Out; ++i) {
      Real z = 0.0;
      for (int j = 0; j < In; ++j) {
        z += W[i * In + j] * in[j];
      }
      z += b[i];
      out[i] = Activation::apply(z);
    }
  }
};

/**
 * @brief Compile-time feedforward network with static forward.
 */
template <typename... Layers> class PINN {
public:
  using Architecture = std::tuple<Layers...>;
  static constexpr int TotalParams = (0 + ... + Layers::NumParams);
  static constexpr int MaxLayerSize = std::max({Layers::InSize..., Layers::OutSize...});

  std::vector<Real> params;

  PINN() {
    params.resize(TotalParams);
    init_params();
  }

  /// @brief Initialize parameters with layer-wise uniform bounds.
  void init_params() {
    std::random_device rd;
    std::mt19937 gen(rd());
    Real *ptr = params.data();

    auto init_layer = [&](auto &&self, auto index_const) -> void {
      constexpr size_t I = decltype(index_const)::value;
      if constexpr (I < sizeof...(Layers)) {
        using LayerType = std::tuple_element_t<I, Architecture>;
        Real limit = std::sqrt(6.0 / (LayerType::InSize + LayerType::OutSize));
        std::uniform_real_distribution<Real> dist(-limit, limit);
        // Fill parameters for the current layer.
        for (int i = 0; i < LayerType::NumParams; ++i)
          *ptr++ = dist(gen);
        self(self, std::integral_constant<size_t, I + 1>{});
      }
    };
    init_layer(init_layer, std::integral_constant<size_t, 0>{});
  }

  /// @brief Apply all layers in sequence using alternating buffers.
  template <size_t I> static ENZYME_INLINE void process_layers(const Real *&p_ptr, Real *input_buf, Real *output_buf) {
    if constexpr (I < sizeof...(Layers)) {
      using CurrentLayer = std::tuple_element_t<I, Architecture>;

      // Layer forward pass.
      CurrentLayer::forward(p_ptr, input_buf, output_buf);

      p_ptr += CurrentLayer::NumParams;

      process_layers<I + 1>(p_ptr, output_buf, input_buf);
    }
  }

  /// @brief Stateless forward evaluation for a single input.
  static ENZYME_INLINE Real forward_static(const Real *x_ptr, const Real *p_ptr) {
    Real buf1[MaxLayerSize];
    Real buf2[MaxLayerSize];

    // Clear scratch buffers.
    for (int i = 0; i < MaxLayerSize; ++i) {
      buf1[i] = 0.0;
      buf2[i] = 0.0;
    }

    using FirstLayer = std::tuple_element_t<0, Architecture>;
    // Load input into the first buffer.
    for (int i = 0; i < FirstLayer::InSize; ++i)
      buf1[i] = x_ptr[i];

    process_layers<0>(p_ptr, buf1, buf2);

    constexpr bool OddLayers = (sizeof...(Layers) % 2 != 0);
    return OddLayers ? buf2[0] : buf1[0];
  }
};
