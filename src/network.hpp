#pragma once
/**
 * @file network.hpp
 * @brief Feedforward network composed of dense layers.
 */

#include "layer.hpp"
#include "seed.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace cpu_mlp {

/**
 * @brief Simple feedforward network with contiguous parameter storage.
 */
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

  /// @brief Total number of parameters.
  size_t getSize() const { return params_size; }

  /// @brief Append a dense layer to the network.
  template <int In, int Out, typename Activation = Linear> void addLayer() {
    layers.push_back(std::make_unique<DenseLayer<In, Out, Activation>>());
    params_size += layers.back()->getParamsSize();
  }

  /// @brief Bind and initialize parameters and gradient buffers.
  void bindParams(unsigned int seed = kDefaultSeed) {
    if (layers.empty()) return;

    // Allocate contiguous storage for parameters and gradients.
    params.resize(params_size);
    grads.resize(params_size);

    std::mt19937 gen(seed);

    activations.clear();
    deltas.clear();

    double *p_ptr = params.data();
    double *g_ptr = grads.data();

    for (auto &layer : layers) {
      // Bind layer views into the shared buffers.
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

  /// @brief Run a forward pass for a batch of inputs.
  const Eigen::MatrixXd &forward(const Eigen::MatrixXd &input) {
    if (activations.empty() || activations[0].cols() != input.cols()) {
      // Resize cached activations for the current batch.
      activations.resize(layers.size() + 1);
    }

    activations[0] = input;

    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->forward(activations[i], activations[i + 1]);
    }

    return activations.back();
  }

  /// @brief Run a backward pass from output loss gradients.
  void backward(const Eigen::MatrixXd &loss_grad) {
    if (deltas.size() != layers.size() + 1) {
      // Resize cached deltas for the current batch.
      deltas.resize(layers.size() + 1);
    }

    deltas.back() = loss_grad;

    for (int i = layers.size() - 1; i >= 0; --i) {
      // Propagate gradients to the previous layer.
      layers[i]->backward(deltas[i + 1], (i > 0) ? &deltas[i] : nullptr);
    }
  }

  /// @brief Zero the gradient buffer.
  void zeroGrads() { std::fill(grads.begin(), grads.end(), 0.0); }

  /// @brief Access raw parameter buffer.
  double *getParamsData() { return params.data(); }
  /// @brief Access raw gradient buffer.
  double *getGradsData() { return grads.data(); }

  /// @brief Replace parameters with a new vector.
  void setParams(const Eigen::VectorXd &new_params) {
    std::copy(new_params.data(), new_params.data() + params_size, params.begin());
  }

  /// @brief Copy gradients to output vector.
  void getGrads(Eigen::VectorXd &out_grads) { std::copy(grads.begin(), grads.end(), out_grads.data()); }

  /// @brief Evaluate accuracy and MSE for a dataset.
  void test(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, std::string label = "Test Results") {
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

    std::cout << "=== " << label << " ===" << std::endl;
    std::cout << "Samples: " << total << std::endl;
    std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
    std::cout << "Total MSE: " << mse << std::endl;
    std::cout << "====================" << std::endl;
  }
};

} // namespace cpu_mlp
