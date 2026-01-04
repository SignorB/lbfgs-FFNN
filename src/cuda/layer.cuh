#pragma once

#include "cublas_handle.cuh"
#include "kernels.cuh"

namespace cuda_mlp {

class CudaDenseLayer {
public:
  CudaDenseLayer(int in, int out, ActivationType act) : in_(in), out_(out), act_(act) {}

  int in() const { return in_; }
  int out() const { return out_; }
  size_t params_size() const { return static_cast<size_t>(out_) * in_ + out_; }

  void bind(double *params, double *grads) {
    params_ptr_ = params;
    grads_ptr_ = grads;
  }

  double init_stddev() const { return activation_scale(act_) * std::sqrt(1.0 / static_cast<double>(in_)); }

  void forward(CublasHandle &handle, const double *input, int batch, double *output) {
    const double alpha = 1.0;
    const double beta = 0.0;

    const double *W = params_ptr_;
    const double *b = params_ptr_ + static_cast<size_t>(out_) * in_;

    cublas_check(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, out_, batch, in_, &alpha, W, out_, input, in_, &beta,
                             output, out_),
                 "cublasDgemm forward");

    launch_add_bias(output, b, out_, batch);
    launch_activation(output, out_ * batch, act_);
  }

  void backward(CublasHandle &handle, const double *input, const double *output, double *next_grad, int batch,
                double *prev_grad) {
    const double alpha = 1.0;
    const double beta = 0.0;

    const double *W = params_ptr_;
    double *dW = grads_ptr_;
    double *db = grads_ptr_ + static_cast<size_t>(out_) * in_;

    launch_activation_deriv(next_grad, output, out_ * batch, act_);

    cublas_check(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, out_, in_, batch, &alpha, next_grad, out_, input, in_,
                             &beta, dW, out_),
                 "cublasDgemm dW");

    launch_sum_rows(next_grad, db, out_, batch);

    if (prev_grad) {
      cublas_check(cublasDgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, in_, batch, out_, &alpha, W, out_, next_grad, out_,
                               &beta, prev_grad, in_),
                   "cublasDgemm dX");
    }
  }

private:
  int in_ = 0;
  int out_ = 0;
  ActivationType act_ = ActivationType::Linear;
  double *params_ptr_ = nullptr;
  double *grads_ptr_ = nullptr;
};

} // namespace cuda_mlp
