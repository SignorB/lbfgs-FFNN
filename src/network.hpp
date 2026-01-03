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

  std::vector<Eigen::MatrixXd> activations;
  std::vector<Eigen::MatrixXd> deltas;

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

    activations.clear();
    deltas.clear();

    double *p_ptr = params.data();
    double *g_ptr = grads.data();

    for (auto &layer : layers) {
      layer->bind(p_ptr, g_ptr);
      double std_dev = layer->getInitStdDev();
      std::normal_distribution<double> dist(0.0, std_dev);
      for (int i = 0; i < layer->getParamsSize(); ++i) {
        p_ptr[i] = dist(gen);
      }
      p_ptr += layer->getParamsSize();
      g_ptr += layer->getParamsSize();
    }
  }

  const Eigen::MatrixXd &forward(const Eigen::MatrixXd &input) {
    if (activations.empty() || activations[0].cols() != input.cols()) {
        activations.resize(layers.size() + 1);
    }
    
    activations[0] = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->forward(activations[i], activations[i + 1]);
    }

    return activations.back();
  }

  void backward(const Eigen::MatrixXd &loss_grad) {
    if (deltas.size() != layers.size() + 1) {
        deltas.resize(layers.size() + 1);
    }

    deltas.back() = loss_grad;

    for (int i = layers.size() - 1; i >= 0; --i) {
      layers[i]->backward(
          deltas[i + 1],
          (i > 0) ? &deltas[i] : nullptr);
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

  void train(const Eigen::MatrixXd &inputs,
             const Eigen::MatrixXd &targets,
             std::shared_ptr<MinimizerBase<Eigen::VectorXd, Eigen::MatrixXd>> minimizer) {

    Eigen::VectorXd x(params_size);
    std::copy(params.begin(), params.end(), x.data());

    VecFun<Eigen::VectorXd, double> f = [&](const Eigen::VectorXd &p) -> double {
      this->setParams(p);
      const auto &output = this->forward(inputs);
      Eigen::MatrixXd diff = output - targets;
      double total_loss = 0.5 * diff.squaredNorm();
      std::cout << "MSE: " << total_loss << std::endl;
      return total_loss;
    };

    GradFun<Eigen::VectorXd> g = [&](const Eigen::VectorXd &p) -> Eigen::VectorXd {
      this->setParams(p);
      this->zeroGrads();

      const auto &output = this->forward(inputs);
      Eigen::MatrixXd loss_grad = output - targets;

      this->backward(loss_grad);

      Eigen::VectorXd grad_vec(params_size);
      this->getGrads(grad_vec);
      return grad_vec;
    };

    Eigen::VectorXd final_params = minimizer->solve(x, f, g);
    this->setParams(final_params);
  }

  void test(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets) {
    const auto &output = this->forward(inputs);
    long correct = 0;
    long total = inputs.cols();

    for (long i = 0; i < total; ++i) {
        Eigen::Index pred_idx, true_idx;
        output.col(i).maxCoeff(&pred_idx);
        targets.col(i).maxCoeff(&true_idx);
        if (pred_idx == true_idx) {
            correct++;
        }
    }

    double accuracy = (double)correct / total * 100.0;
    Eigen::MatrixXd diff = output - targets;
    double mse = 0.5 * diff.squaredNorm();

    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Samples: " << total << std::endl;
    std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
    std::cout << "Total MSE: " << mse << std::endl;
    std::cout << "====================" << std::endl;
  }
};
