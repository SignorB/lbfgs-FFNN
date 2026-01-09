#pragma once
#include "common.hpp"
#include <Eigen/Core>
#include <Eigen/src/Core/Map.h>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>


//implementation for PINN
//removed derivative functions
//modified to template on type T for autodiff compatibility


struct Linear {
template <typename T>
  static inline T apply(T x) { return x; }

  static constexpr double scale = 1.0;
};


struct ReLU {

template <typename T>
  static inline T apply(T x) { return (x > 0.0) ? x : 0.0; }

  static constexpr double scale = 1.41421356;
};


struct Sigmoid {
    
template <typename T>
  static inline T apply(T x) { return 1.0 / (1.0 + std::exp(-x)); }
  
  static constexpr double scale = 1.0;
};



struct Tanh {
    template <typename T>
  static inline T apply(T x) { 
    using std::tanh;
    return tanh(x); }
  static constexpr double scale = 1.0;
};

template <typename T>
class Layer {
public:

  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  virtual ~Layer() = default;

  virtual void bind(T *params) = 0;
  virtual void forward(const MatrixT &input, MatrixT &output) = 0;
  
  virtual int getInSize() const = 0;
  virtual int getOutSize() const = 0;
  virtual int getParamsSize() const = 0;
  virtual double getInitStdDev() const = 0;
};

// in = number of cols (input), out = number of rows (output)
template <typename T, int In, int Out, typename Activation = Linear>
class DenseLayer : public Layer<T> {
private:

  using MatrixT = typename Layer<T>::MatrixT;
  using VectorT = typename Layer<T>::VectorT;


  //maps to read weights as matrix and bias as vector
  using MapMatW = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
  using MapVecB = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;


  T *params_ptr = nullptr;
 
public:
  DenseLayer() {}

  int getInSize() const override { return In; }
  int getOutSize() const override { return Out; }
  int getParamsSize() const override { return (Out * In) + Out; }

    void bind(T *params) override {
        params_ptr = params;
    }

  void forward(const MatrixT &input, MatrixT &output) override {
    MapMatW W(params_ptr, Out, In);
    MapVecB b(params_ptr + (Out * In), Out);

    MatrixT z = W * input;
    z.colwise() += b;

    output = z.unaryExpr([](T v) {
      return Activation::apply(v);
    });
  }

  double getInitStdDev() const override {
    return Activation::scale * std::sqrt(1.0 / (double)In);
  }
};
