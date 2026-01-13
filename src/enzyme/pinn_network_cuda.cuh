#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <tuple>

// Heavy inlining keeps Enzyme's device AD from losing track of intermediates.
#ifndef PINN_CUDA_INLINE
#define PINN_CUDA_INLINE __device__ __forceinline__
#endif

// Small helper to compute a compile-time maximum over a parameter pack.
template <int First, int... Rest>
struct StaticMax {
    static constexpr int value = First > StaticMax<Rest...>::value ? First : StaticMax<Rest...>::value;
};

template <int Last>
struct StaticMax<Last> {
    static constexpr int value = Last;
};

// Enzyme CUDA intrinsics: single fixed prototype (device varargs unsupported).
extern "C" {
__device__ double __enzyme_autodiff(double (*fn)(const void *, const void *),
    int, const void *, void *, int, const void *);
__device__ int enzyme_dup;
__device__ int enzyme_const;
}

namespace pinn_cuda {

// Simple activations.
struct Tanh {
    PINN_CUDA_INLINE static double apply(double x) { return tanh(x); }
};

struct Linear {
    PINN_CUDA_INLINE static double apply(double x) { return x; }
};

template <int In, int Out, typename Activation>
struct Dense {
    static constexpr int InSize = In;
    static constexpr int OutSize = Out;
    static constexpr int NumParams = (In * Out) + Out;

    PINN_CUDA_INLINE static void forward(const double *p, const double *in, double *out) {
        const double *W = p;
        const double *b = p + (In * Out);

        #pragma unroll
        for (int i = 0; i < Out; ++i) {
            double z = 0.0;
            #pragma unroll
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
    static constexpr int MaxLayerSize = StaticMax<Layers::InSize..., Layers::OutSize...>::value;

    double params[TotalParams];

    __host__ PINN() { init_params(); }

    __host__ void init_params() {
        // Lightweight deterministic init (no std::random) to keep dependencies minimal.
        unsigned int state = 17u;
        for (int i = 0; i < TotalParams; ++i) {
            state = state * 1664525u + 1013904223u;
            double r = (static_cast<double>((state >> 8) & 0xFFFFFFu) / 0xFFFFFFu) - 0.5;
            params[i] = 0.2 * r;  // keep weights small
        }
    }

    template <size_t I>
    PINN_CUDA_INLINE static void process_layers(const double *&p_ptr, double *input_buf, double *output_buf) {
        if constexpr (I < sizeof...(Layers)) {
            using CurrentLayer = std::tuple_element_t<I, Architecture>;
            CurrentLayer::forward(p_ptr, input_buf, output_buf);
            p_ptr += CurrentLayer::NumParams;
            process_layers<I + 1>(p_ptr, output_buf, input_buf);
        }
    }

    PINN_CUDA_INLINE static double forward_primal(const double *x_ptr, const double *p_ptr) {
        double buf1[MaxLayerSize];
        double buf2[MaxLayerSize];

        #pragma unroll
        for (int i = 0; i < MaxLayerSize; ++i) {
            buf1[i] = 0.0;
            buf2[i] = 0.0;
        }

        using FirstLayer = std::tuple_element_t<0, Architecture>;
        #pragma unroll
        for (int i = 0; i < FirstLayer::InSize; ++i) buf1[i] = x_ptr[i];

        const double *walker = p_ptr;
        process_layers<0>(walker, buf1, buf2);

        constexpr bool OddLayers = (sizeof...(Layers) % 2 != 0);
        return OddLayers ? buf2[0] : buf1[0];
    }

    // Adapter with generic pointer signature for Enzyme.
    PINN_CUDA_INLINE static double forward_device(const void *x_void, const void *p_void) {
        const double *x_ptr = static_cast<const double *>(x_void);
        const double *p_ptr = static_cast<const double *>(p_void);
        return forward_primal(x_ptr, p_ptr);
    }
};

// ---- Burgers-specific helpers ----

constexpr double PINN_PI = 3.14159265358979323846;

template <typename Net>
PINN_CUDA_INLINE double diff_input(const double *xt_ptr, const double *p, int index) {
    double input_grad[2] = {0.0, 0.0};
    __enzyme_autodiff(Net::forward_device, enzyme_dup, xt_ptr, input_grad, enzyme_const, p);
    return input_grad[index];
}

template <typename Net>
PINN_CUDA_INLINE double calc_ux(const double *xt_ptr, const double *p) {
    return diff_input<Net>(xt_ptr, p, 0);
}

template <typename Net>
PINN_CUDA_INLINE double calc_ux_adapter(const void *xt, const void *params) {
    return calc_ux<Net>(static_cast<const double *>(xt), static_cast<const double *>(params));
}

template <typename Net>
PINN_CUDA_INLINE double burgers_residual(const double *xt_ptr, const double *p) {
    const double nu = 0.3 / PINN_PI;
    double u = Net::forward_primal(xt_ptr, p);
    double ut = diff_input<Net>(xt_ptr, p, 1);
    double ux = diff_input<Net>(xt_ptr, p, 0);

    double grad_ux[2] = {0.0, 0.0};
    __enzyme_autodiff(calc_ux_adapter<Net>, enzyme_dup, xt_ptr, grad_ux, enzyme_const, p);
    double uxx = grad_ux[0];

    return ut + u * ux - nu * uxx;
}

enum SampleType { Initial = 0, Boundary = 1, Collocation = 2 };

struct BurgersData {
    const double *xs;
    const double *ts;
    const int *kind;
    int count;
};

template <typename Net>
PINN_CUDA_INLINE double burgers_loss(const void *p_void, const void *data_void) {
    const double *p = static_cast<const double *>(p_void);
    const BurgersData *data = static_cast<const BurgersData *>(data_void);

    const double *xs = data->xs;
    const double *ts = data->ts;
    const int *kind = data->kind;
    const int count = data->count;

    double loss_ic = 0.0;
    double loss_bc = 0.0;
    double loss_pde = 0.0;
    int n_ic = 0, n_bc = 0, n_pde = 0;

    for (int i = 0; i < count; ++i) {
        double xt[2] = {xs[i], ts[i]};
        int k = kind[i];
        if (k == SampleType::Initial) {
            double u = Net::forward_primal(xt, p);
            double target = sin(PINN_PI * xt[0]);
            double diff = u - target;
            loss_ic += diff * diff;
            ++n_ic;
        } else if (k == SampleType::Boundary) {
            double u = Net::forward_primal(xt, p);
            loss_bc += u * u;
            ++n_bc;
        } else {
            double r = burgers_residual<Net>(xt, p);
            loss_pde += r * r;
            ++n_pde;
        }
    }

    if (n_ic > 0) loss_ic /= static_cast<double>(n_ic);
    if (n_bc > 0) loss_bc /= static_cast<double>(n_bc);
    if (n_pde > 0) loss_pde /= static_cast<double>(n_pde);

    const double w_ic = 1.0;
    const double w_bc = 2.0;
    const double w_pde = 4.0;

    return w_bc * loss_bc + w_ic * loss_ic + w_pde * loss_pde;
}

template <typename Net>
__global__ void burgers_loss_and_grad(const double *xs,
    const double *ts,
    const int *kind,
    int count,
    const double *p,
    double *grad_out,
    double *loss_out) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        BurgersData data{xs, ts, kind, count};
        double l_primal = burgers_loss<Net>(p, &data);
        __enzyme_autodiff(
            burgers_loss<Net>,
            enzyme_dup, p, grad_out,
            enzyme_const, &data);
        if (loss_out) loss_out[0] = l_primal;
    }
}

}  // namespace pinn_cuda
