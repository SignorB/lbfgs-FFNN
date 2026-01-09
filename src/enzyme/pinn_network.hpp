#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <array>
#include <random>
#include <algorithm> 

/*
macro needed to force inling via compiler. In fact inline is not always
respected by our silly friend
 */
#ifndef ENZYME_INLINE
#define ENZYME_INLINE __attribute__((always_inline)) inline
#endif

struct Tanh {
    static ENZYME_INLINE double apply(double x) { return std::tanh(x); }
};

struct Linear {
    static ENZYME_INLINE double apply(double x) { return x; }
};


template <int In, int Out, typename Activation>
struct Dense {
    static constexpr int InSize = In;
    static constexpr int OutSize = Out;
    static constexpr int NumParams = (In * Out) + Out;

    static ENZYME_INLINE void forward(const double* p, const double* in, double* out) {
        const double* W = p;
        const double* b = p + (In * Out);

        for (int i = 0; i < Out; ++i) {
            double z = 0.0;
            for (int j = 0; j < In; ++j) {
                z += W[i * In + j] * in[j];
            }
            z += b[i];
            out[i] = Activation::apply(z);
        }
    }
};

template <typename... Layers>
class PINN {
public:
    using Architecture = std::tuple<Layers...>;
    static constexpr int TotalParams = (0 + ... + Layers::NumParams);
    static constexpr int MaxLayerSize = std::max({Layers::InSize..., Layers::OutSize...});

    std::vector<double> params;

    PINN() {
        params.resize(TotalParams);
        init_params();
    }

    void init_params() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double* ptr = params.data();
        
        auto init_layer = [&](auto&& self, auto index_const) -> void {
            constexpr size_t I = decltype(index_const)::value;
            if constexpr (I < sizeof...(Layers)) {
                using LayerType = std::tuple_element_t<I, Architecture>;
                double limit = std::sqrt(6.0 / (LayerType::InSize + LayerType::OutSize));
                std::uniform_real_distribution<double> dist(-limit, limit);
                for(int i=0; i < LayerType::NumParams; ++i) *ptr++ = dist(gen);
                self(self, std::integral_constant<size_t, I + 1>{});
            }
        };
        init_layer(init_layer, std::integral_constant<size_t, 0>{});
    }

    template <size_t I>
    static ENZYME_INLINE void process_layers(const double*& p_ptr, double* input_buf, double* output_buf) {
        if constexpr (I < sizeof...(Layers)) {
            using CurrentLayer = std::tuple_element_t<I, Architecture>;
            
            CurrentLayer::forward(p_ptr, input_buf, output_buf);
            
            p_ptr += CurrentLayer::NumParams;

            process_layers<I + 1>(p_ptr, output_buf, input_buf);
        }
    }


    static ENZYME_INLINE double forward_static(const double* x_ptr, const double* p_ptr) {
        double buf1[MaxLayerSize];
        double buf2[MaxLayerSize];

        for(int i=0; i<MaxLayerSize; ++i) { buf1[i] = 0.0; buf2[i] = 0.0; }

        using FirstLayer = std::tuple_element_t<0, Architecture>;
        for(int i=0; i<FirstLayer::InSize; ++i) buf1[i] = x_ptr[i];

        process_layers<0>(p_ptr, buf1, buf2);

        constexpr bool OddLayers = (sizeof...(Layers) % 2 != 0);
        return OddLayers ? buf2[0] : buf1[0];
    }
};
