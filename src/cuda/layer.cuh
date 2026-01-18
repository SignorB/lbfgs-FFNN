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
  size_t weights_size() const { return static_cast<size_t>(out_) * in_; }
  size_t bias_size() const { return static_cast<size_t>(out_); }

  void bind(CudaScalar *params, CudaScalar *grads) {
    params_ptr_ = params;
    grads_ptr_ = grads;
  }

  CudaScalar init_stddev() const {
    return activation_scale(act_) * std::sqrt(CudaScalar{1.0f} / static_cast<CudaScalar>(in_));
  }

  void forward(CublasHandle &handle, const CudaScalar *input, int batch, CudaScalar *output) {
    const CudaScalar alpha = 1.0f, beta = 0.0f, *W = params_ptr_, *b = params_ptr_ + static_cast<size_t>(out_) * in_;

    cublas_check(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, out_, batch, in_, &alpha, W, out_, input, in_, &beta,
                             output, out_),
                 "cublasSgemm forward");

    launch_add_bias(output, b, out_, batch);
    launch_activation(output, out_ * batch, act_);
  }

  void backward(CublasHandle &handle, const CudaScalar *input, const CudaScalar *output, CudaScalar *next_grad, int batch,
                CudaScalar *prev_grad) {
    const CudaScalar alpha = 1.0f, beta = 0.0f, *W = params_ptr_;

    CudaScalar *dW = grads_ptr_, *db = grads_ptr_ + static_cast<size_t>(out_) * in_;

    launch_activation_deriv(next_grad, output, out_ * batch, act_);

    cublas_check(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, out_, in_, batch, &alpha, next_grad, out_, input, in_,
                             &beta, dW, out_),
                 "cublasSgemm dW");

    launch_sum_rows(next_grad, db, out_, batch);

    if (prev_grad) {
      cublas_check(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, in_, batch, out_, &alpha, W, out_, next_grad, out_,
                               &beta, prev_grad, in_), "cublasSgemm dX");
    }
  }

private:
  int in_ = 0, out_ = 0;
  ActivationType act_ = ActivationType::Linear;
  CudaScalar *params_ptr_ = nullptr, *grads_ptr_ = nullptr;

public:
  const CudaScalar *params_ptr() const { return params_ptr_; }
  const CudaScalar *grads_ptr() const { return grads_ptr_; }
};

} // namespace cuda_mlp
