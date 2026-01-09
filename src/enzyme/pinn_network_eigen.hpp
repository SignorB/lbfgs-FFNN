// DO NOT USE FOR NOW DOESN'T WORK 
#pragma once

#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_ALIGN_STATICALLY
#define EIGEN_NO_DEBUG

#include <vector>
#include <cmath>
#include <tuple>
#include <array>
#include <random>
#include <algorithm>
#include <cstdlib> 
#include <Eigen/Core>

#ifndef ENZYME_INLINE
#define ENZYME_INLINE __attribute__((always_inline)) inline
#endif

struct Tanh {
    static ENZYME_INLINE double apply(double x) { return std::tanh(x); }
};

struct Linear {
    static ENZYME_INLINE double apply(double x) { return x; }
};

struct Sigmoid {
    static ENZYME_INLINE double apply(double x) { return 1.0 / (1.0 + std::exp(-x)); }
};

template <int In, int Out, typename Activation>
struct Dense {
    static constexpr int InSize = In;
    static constexpr int OutSize = Out;
    static constexpr int NumParams = (In * Out) + Out;

    using WeightsMat = Eigen::Matrix<double, Out, In>; 
    using BiasVec    = Eigen::Matrix<double, Out, 1>;
    
    using MapWeights = Eigen::Map<const WeightsMat>;
    using MapBias    = Eigen::Map<const BiasVec>;
    using MapInput   = Eigen::Map<const Eigen::Matrix<double, In, 1>>;
    using MapOutput  = Eigen::Map<Eigen::Matrix<double, Out, 1>>;

    static ENZYME_INLINE void forward(const double* p, const double* in, double* out) {
        MapWeights W(p);
        MapBias    b(p + (In * Out));
        MapInput   x(in);
        MapOutput  y(out);

        y.noalias() = (W * x) + b;

        for (int i = 0; i < Out; ++i) {
            out[i] = Activation::apply(out[i]);
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
                double limit = std::sqrt(6.0 / (double)(LayerType::InSize + LayerType::OutSize));
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
        double* buf1 = (double*)malloc(MaxLayerSize * sizeof(double));
        double* buf2 = (double*)malloc(MaxLayerSize * sizeof(double));

        using FirstLayer = std::tuple_element_t<0, Architecture>;
        for(int i=0; i<FirstLayer::InSize; ++i) buf1[i] = x_ptr[i];

        process_layers<0>(p_ptr, buf1, buf2);

        constexpr bool OddLayers = (sizeof...(Layers) % 2 != 0);
        double result = OddLayers ? buf2[0] : buf1[0];

        free(buf1);
        free(buf2);

        return result;
    }
};
