#pragma once
#include "layer_pinn.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include "minimizer_base.hpp"
#include "s_lbfgs.hpp"
#include <iostream>

//this only has to work as a function approximation network
//so no backward pass is needed
template <typename T>
class Network {

  public:
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Network() = default;

  size_t getSize() const {
    return params_size;
  }

  template <int In, int Out, typename Activation = Linear>
  void addLayer() {
    layers.push_back(std::make_unique<DenseLayer<T, In, Out, Activation>>());
    params_size += layers.back()->getParamsSize();
  }

  void bindParams() {
    if (layers.empty())
      return;

    params.resize(params_size);
   // grads.resize(params_size);

    std::random_device rd;
    std::mt19937 gen(rd());

    // activations.clear();
   // deltas.clear();

    T *p_ptr = params.data();
  //  double *g_ptr = grads.data();

    for (auto &layer : layers) {
      layer->bind(p_ptr);
      double std_dev = layer->getInitStdDev();
      std::normal_distribution<double> dist(0.0, std_dev);
      for (int i = 0; i < layer->getParamsSize(); ++i) {
        p_ptr[i] = dist(gen);
      }
      p_ptr += layer->getParamsSize();
    //  g_ptr += layer->getParamsSize();
    }
  }

  MatrixT forward(const MatrixT &input) {
    MatrixT current = input;
    MatrixT next;
    
    for (auto &layer : layers) {
      layer->forward(current, next);
      current = next;
    }
    return current;
  }


  void setParams(const VectorT &new_params) {
    std::copy(new_params.data(), new_params.data() + params_size, params.begin());
  }


  VectorT getParams() const {
    return Eigen::Map<VectorT>(params.data(), params_size);
  }

  private:
    std::vector<std::unique_ptr<Layer<T>>> layers;
    std::vector<T> params; 

    size_t params_size = 0;
};