#pragma once
#include "common.hpp"
#include <Eigen/Core>
#include <Eigen/src/Core/Map.h>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

namespace cpu_mlp {

struct Linear {
  static inline double apply(double x) { return x; }
  static inline double prime(double /*x*/) { return 1.0; }
  static constexpr double scale = 1.0;
};

struct ReLU {
  static inline double apply(double x) { return (x > 0.0) ? x : 0.0; }
  static inline double prime(double x) { return (x > 0.0) ? 1.0 : 0.0; }
  static constexpr double scale = 1.41421356;
};

struct Sigmoid {
  static inline double apply(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  static inline double prime(double x) {
    double s = apply(x);
    return s * (1.0 - s);
  }
  static constexpr double scale = 1.0;
};

struct Tanh {
  static inline double apply(double x) { return std::tanh(x); }
  static inline double prime(double x) {
    double t = std::tanh(x);
    return 1.0 - (t * t);
  }
  static constexpr double scale = 1.0;
};

class Layer {
public:
  virtual ~Layer() = default;
  virtual void bind(double *params, double *grads) = 0;
  virtual void forward(const Eigen::MatrixXd &input, Eigen::MatrixXd &output) = 0;
  virtual void backward(const Eigen::MatrixXd &next_grad, Eigen::MatrixXd *prev_grad) = 0;
  virtual int getInSize() const = 0;
  virtual int getOutSize() const = 0;
  virtual int getParamsSize() const = 0;
  virtual double getInitStdDev() const = 0;
};

template <int In, int Out, typename Activation = Linear> class DenseLayer : public Layer {
private:
  using MapMatW = Eigen::Map<const Eigen::MatrixXd>;
  using MapVecB = Eigen::Map<const Eigen::VectorXd>;

  using MapMatW_Grad = Eigen::Map<Eigen::MatrixXd>;
  using MapVecB_Grad = Eigen::Map<Eigen::VectorXd>;

  double *params_ptr = nullptr;
  double *grads_ptr = nullptr;

  Eigen::MatrixXd input_cache;
  Eigen::MatrixXd z_cache;

public:
  DenseLayer() {}

  int getInSize() const override { return In; }
  int getOutSize() const override { return Out; }
  int getParamsSize() const override { return (Out * In) + Out; }

  void bind(double *params, double *grads) override {
    params_ptr = params;
    grads_ptr = grads;
  }

  void forward(const Eigen::MatrixXd &input, Eigen::MatrixXd &output) override {
    MapMatW W(params_ptr, Out, In);
    MapVecB b(params_ptr + (Out * In), Out);

    input_cache = input;
    z_cache = W * input;
    z_cache.colwise() += b;

    output = z_cache.unaryExpr([](double v) { return Activation::apply(v); });
  }

  void backward(const Eigen::MatrixXd &next_grad, Eigen::MatrixXd *prev_grad) override {
    MapMatW_Grad dW(grads_ptr, Out, In);
    MapVecB_Grad db(grads_ptr + (Out * In), Out);

    Eigen::MatrixXd dZ = next_grad.cwiseProduct(z_cache.unaryExpr([](double v) { return Activation::prime(v); }));

    // nalias ~= __restrict__
    dW.noalias() += dZ * input_cache.transpose();
    db.noalias() += dZ.rowwise().sum();

    if (prev_grad) {
      MapMatW W(params_ptr, Out, In);
      *prev_grad = W.transpose() * dZ;
    }
  }

  double getInitStdDev() const override { return Activation::scale * std::sqrt(1.0 / (double)In); }
};

} // namespace cpu_mlp
