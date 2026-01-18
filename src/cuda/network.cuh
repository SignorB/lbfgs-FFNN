#pragma once

#include "cublas_handle.cuh"
#include "device_buffer.cuh"
#include "kernels.cuh"
#include "layer.cuh"
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace cuda_mlp {

class CudaNetwork {
public:
  explicit CudaNetwork(CublasHandle &handle) : handle_(handle) {}

  void addLayer(int in, int out, ActivationType act) {
    layers_.emplace_back(in, out, act);
    params_size_ += layers_.back().params_size();
  }

  void bindParams(unsigned int seed = 2341) {
    params_.resize(params_size_);
    grads_.resize(params_size_);

    std::vector<CudaScalar> host_params(params_size_);
    std::mt19937 gen(seed);

    size_t offset = 0;
    for (auto &layer : layers_) {
      layer.bind(params_.data() + offset, grads_.data() + offset);
      CudaScalar std_dev = layer.init_stddev();
      std::normal_distribution<CudaScalar> dist(0.0f, std_dev);
      size_t weights_count = layer.out() * layer.in();
      size_t bias_count = layer.out();
      for (size_t i = 0; i < weights_count; ++i)
        host_params[offset + i] = dist(gen);
      for (size_t i = 0; i < bias_count; ++i)
        host_params[offset + weights_count + i] = 0.0f;
      offset += layer.params_size();
    }

    params_.copy_from_host(host_params.data(), host_params.size());
    zeroGrads();
  }

  size_t params_size() const { return params_size_; }
  int output_size() const { return layers_.empty() ? 0 : layers_.back().out(); }

  CudaScalar *params_data() { return params_.data(); }
  CudaScalar *grads_data() { return grads_.data(); }

  void zeroGrads() { device_set_zero(grads_.data(), grads_.size()); }

  void forward_only(const CudaScalar *input, int batch) {
    ensure_buffers(batch);
    const CudaScalar *current = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
      CudaScalar *out = activations_[i].data();
      layers_[i].forward(handle_, current, batch, out);
      current = out;
    }
    last_batch_ = batch;
  }

  CudaScalar compute_loss_and_grad(const CudaScalar *input, const CudaScalar *target, int batch) {
    forward_only(input, batch);

    CudaScalar *diff = deltas_.back().data();
    int out_size = output_size();
    int total = out_size * batch;
    launch_diff(activations_.back().data(), target, diff, total);

    CudaScalar sum_sq_error = 0.5f * device_dot(handle_, diff, diff, total),
               loss = sum_sq_error / static_cast<CudaScalar>(batch), alpha = 1.0f / static_cast<CudaScalar>(batch);
    device_scal(handle_, total, alpha, diff);

    zeroGrads();
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
      const CudaScalar *layer_input = (i == 0) ? input : activations_[i - 1].data();
      const CudaScalar *layer_output = activations_[i].data();
      CudaScalar *next_grad = deltas_[i].data();
      CudaScalar *prev_grad = (i > 0) ? deltas_[i - 1].data() : nullptr;
      layers_[i].backward(handle_, layer_input, layer_output, next_grad, batch, prev_grad);
    }

    return loss;
  }
  void copy_output_to_host(CudaScalar *host, size_t n) const {
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
  DeviceBuffer<CudaScalar> params_, grads_;
  size_t params_size_ = 0;

  std::vector<DeviceBuffer<CudaScalar>> activations_, deltas_;
  int last_batch_ = 0;
};

} // namespace cuda_mlp
