#pragma once
#include "layer.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include "minimizer_base.hpp"
#include <iostream>

class Network {
private:
  std::vector<std::unique_ptr<Layer>> layers;

  std::vector<double> params;
  std::vector<double> grads;
  size_t params_size = 0;

  std::vector<std::vector<double>> activations;
  std::vector<std::vector<double>> deltas;

public:
  Network() = default;

  size_t getSize() const {
    return params_size;
  }

  template <int In, int Out, typename Activation = Linear>
  void addLayer() {
    layers.push_back(std::make_unique<DenseLayer<In, Out, Activation>>());
    params_size += layers.back()->getParamsSize();
  }

  void bindParams() {
    if (layers.empty())
      return;

    params.resize(params_size);
    grads.resize(params_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.5);

    for (double &p : params) {
      p = dist(gen);
    }

    activations.clear();
    deltas.clear();

    activations.emplace_back(std::vector<double>(layers.front()->getInSize()));
    deltas.emplace_back(std::vector<double>(layers.front()->getInSize()));

    for (const auto &layer : layers) {
      size_t out_size = layer->getOutSize();
      activations.emplace_back(std::vector<double>(out_size));
      deltas.emplace_back(std::vector<double>(out_size));
    }

    double *p_ptr = params.data();
    double *g_ptr = grads.data();

    for (auto &layer : layers) {
      layer->bind(p_ptr, g_ptr);
      p_ptr += layer->getParamsSize();
      g_ptr += layer->getParamsSize();
    }
  }

  const std::vector<double> &forward(const std::vector<double> &input) {
    activations[0] = input;

    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->forward(activations[i].data(), activations[i + 1].data());
    }

    return activations.back();
  }

  void backward(const std::vector<double> &loss_grad) {
    deltas.back() = loss_grad;

    for (int i = layers.size() - 1; i >= 0; --i) {
      layers[i]->backward(
          deltas[i + 1].data(),
          deltas[i].data());
    }
  }

  void zeroGrads() {
    std::fill(grads.begin(), grads.end(), 0.0);
  }

  double *getParamsData() { return params.data(); }
  double *getGradsData() { return grads.data(); }

  void setParams(const Eigen::VectorXd &new_params) {
    std::copy(new_params.data(), new_params.data() + params_size, params.begin());
  }

  void getGrads(Eigen::VectorXd &out_grads) {
    std::copy(grads.begin(), grads.end(), out_grads.data());
  }

  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets,
             std::shared_ptr<MinimizerBase<Eigen::VectorXd, Eigen::MatrixXd>> minimizer) {


    Eigen::VectorXd x(params_size);
    std::copy(params.begin(), params.end(), x.data());

    VecFun<Eigen::VectorXd, double> f = [&](const Eigen::VectorXd &p) -> double {
      this->setParams(p);
      double total_loss = 0.0;

      for (size_t k = 0; k < inputs.size(); ++k) {
        const auto &output = this->forward(inputs[k]);
        for (size_t i = 0; i < output.size(); ++i) {
          double err = output[i] - targets[k][i];
          total_loss += 0.5 * err * err;
        }
      }
      std::cout << "MSE: " << total_loss << std::endl;
      return total_loss;
    };

    GradFun<Eigen::VectorXd> g = [&](const Eigen::VectorXd &p) -> Eigen::VectorXd {
      this->setParams(p);
      this->zeroGrads();

      std::vector<double> loss_grad(layers.back()->getOutSize());

      for (size_t k = 0; k < inputs.size(); ++k) {
        const auto &output = this->forward(inputs[k]);

        for (size_t i = 0; i < output.size(); ++i) {
          loss_grad[i] = output[i] - targets[k][i];
        }

        this->backward(loss_grad);
      }

      Eigen::VectorXd grad_vec(params_size);
      this->getGrads(grad_vec);
      return grad_vec;
    };

    Eigen::VectorXd final_params = minimizer->solve(x, f, g);
    this->setParams(final_params);
  }
};
