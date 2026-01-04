#pragma once

#include "cublas_handle.cuh"
#include "device_buffer.cuh"
#include "kernels.cuh"
#include "layer.cuh"
#include <random>
#include <vector>

namespace cuda_mlp {

class CudaNetwork {
public:
  explicit CudaNetwork(CublasHandle &handle) : handle_(handle) {}

  void addLayer(int in, int out, ActivationType act) {
    layers_.emplace_back(in, out, act);
    params_size_ += layers_.back().params_size();
  }

  void bindParams(unsigned int seed = 42) {
    params_.resize(params_size_);
    grads_.resize(params_size_);

    std::vector<double> host_params(params_size_);
    std::mt19937 gen(seed);

    size_t offset = 0;
    for (auto &layer : layers_) {
      layer.bind(params_.data() + offset, grads_.data() + offset);
      double std_dev = layer.init_stddev();
      std::normal_distribution<double> dist(0.0, std_dev);
      for (size_t i = 0; i < layer.params_size(); ++i) {
        host_params[offset + i] = dist(gen);
      }
      offset += layer.params_size();
    }

    params_.copy_from_host(host_params.data(), host_params.size());
    zeroGrads();
  }

  size_t params_size() const { return params_size_; }
  int output_size() const { return layers_.empty() ? 0 : layers_.back().out(); }

  double *params_data() { return params_.data(); }
  double *grads_data() { return grads_.data(); }

  void zeroGrads() { device_set_zero(grads_.data(), grads_.size()); }

  void forward_only(const double *input, int batch) {
    ensure_buffers(batch);
    const double *current = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
      double *out = activations_[i].data();
      layers_[i].forward(handle_, current, batch, out);
      current = out;
    }
    last_batch_ = batch;
  }

  double compute_loss_and_grad(const double *input, const double *target, int batch) {
    forward_only(input, batch);

    double *diff = deltas_.back().data();
    int out_size = output_size();
    int total = out_size * batch;
    launch_diff(activations_.back().data(), target, diff, total);

    double loss = 0.5 * device_dot(handle_, diff, diff, total);

    zeroGrads();
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      const double *layer_input = (i == 0) ? input : activations_[i - 1].data();
      const double *layer_output = activations_[i].data();
      double *next_grad = deltas_[i].data();
      double *prev_grad = (i > 0) ? deltas_[i - 1].data() : nullptr;
      layers_[i].backward(handle_, layer_input, layer_output, next_grad, batch, prev_grad);
    }

    return loss;
  }

  void copy_output_to_host(double *host, size_t n) const {
    if (activations_.empty()) {
      return;
    }
    activations_.back().copy_to_host(host, n);
  }

  int last_batch() const { return last_batch_; }

private:
  void ensure_buffers(int batch) {
    if (batch == last_batch_ && activations_.size() == layers_.size()) {
      return;
    }

    activations_.clear();
    deltas_.clear();
    activations_.reserve(layers_.size());
    deltas_.reserve(layers_.size());

    for (const auto &layer : layers_) {
      activations_.emplace_back(static_cast<size_t>(layer.out()) * batch);
      deltas_.emplace_back(static_cast<size_t>(layer.out()) * batch);
    }
  }

  CublasHandle &handle_;
  std::vector<CudaDenseLayer> layers_;
  DeviceBuffer<double> params_;
  DeviceBuffer<double> grads_;
  size_t params_size_ = 0;

  std::vector<DeviceBuffer<double>> activations_;
  std::vector<DeviceBuffer<double>> deltas_;
  int last_batch_ = 0;
};

} // namespace cuda_mlp
