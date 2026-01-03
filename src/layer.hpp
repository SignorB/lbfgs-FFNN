#pragma once
#include "common.hpp"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/src/Core/Map.h>

struct Linear {
    static inline double apply(double x) { return x; }
    static inline double prime(double x) { return 1.0; }
};

struct ReLU {
    static inline double apply(double x) { return (x > 0.0) ? x : 0.0; }
    static inline double prime(double x) { return (x > 0.0) ? 1.0 : 0.0; }
};

struct Sigmoid {
    static inline double apply(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static inline double prime(double x) {
        double s = apply(x);
        return s * (1.0 - s);
    }
};

struct Tanh {
    static inline double apply(double x) { return std::tanh(x); }
    static inline double prime(double x) {
        double t = std::tanh(x);
        return 1.0 - (t * t);
    }
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual void bind(double* params, double* grads) = 0;
    virtual void forward(const double* input, double* output) = 0;
    virtual void backward(const double* next_grad, double* prev_grad) = 0;
    virtual int getInSize() const = 0;
    virtual int getOutSize() const = 0;
    virtual int getParamsSize() const = 0;
};

template <int In, int Out, typename Activation = Linear>
class DenseLayer : public Layer {
private:
    using VecIn   = Eigen::Matrix<double, In, 1>;
    using VecOut  = Eigen::Matrix<double, Out, 1>;

    using MapMatW = Eigen::Map<const Eigen::Matrix<double, Out, In>>;
    using MapVecB = Eigen::Map<const Eigen::Matrix<double, Out, 1>>;
    
    using MapMatW_Grad = Eigen::Map<Eigen::Matrix<double, Out, In>>;
    using MapVecB_Grad = Eigen::Map<Eigen::Matrix<double, Out, 1>>;

    double* params_ptr = nullptr;
    double* grads_ptr = nullptr;

    VecIn  input_cache;
    VecOut z_cache;

public:

    DenseLayer() {}

    int getInSize() const override { return In; }
    int getOutSize() const override { return Out; }
    int getParamsSize() const override { return (Out * In) + Out; }

    void bind(double* params, double* grads) override {
        params_ptr = params;
        grads_ptr = grads;
    }

    void forward(const double* input, double* output) override {
        Eigen::Map<const VecIn> x(input);
        Eigen::Map<VecOut> y(output);
        MapMatW W(params_ptr);
        MapVecB b(params_ptr + (Out * In));

        z_cache = W * x + b;
        input_cache = x;

        y = z_cache.unaryExpr([](double v) { 
            return Activation::apply(v); 
        });
    }

    void backward(const double* next_grad_ptr, double* prev_grad_ptr) override {
        Eigen::Map<const VecOut> delta_next(next_grad_ptr);
        MapMatW_Grad dW(grads_ptr);
        MapVecB_Grad db(grads_ptr + (Out * In));

        VecOut dZ = delta_next.cwiseProduct(
            z_cache.unaryExpr([](double v) { return Activation::prime(v); })
        );

        dW += dZ * input_cache.transpose();
        db += dZ;

        if (prev_grad_ptr) {
            MapMatW W(params_ptr);
            Eigen::Map<Eigen::Matrix<double, In, 1>> delta_prev(prev_grad_ptr);
            delta_prev = W.transpose() * dZ;
        }
    }
};
