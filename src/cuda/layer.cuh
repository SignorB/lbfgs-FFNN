#pragma once

#include "cublas_handle.cuh"
#include "kernels.cuh"

namespace cuda_mlp {

/// @brief Fully connected layer with activation, using column-major matrices
class CudaDenseLayer {
public:
  /**
   * @brief Construct a dense layer
   * @param in Input dimension
   * @param out Output dimension
   * @param act Activation type
   */
  CudaDenseLayer(int in, int out, ActivationType act) : in_(in), out_(out), act_(act) {}

  /// @brief Input dimension
  int in() const { return in_; }
  /// @brief Output dimension
  int out() const { return out_; }
  /// @brief Total parameter count (weights + bias)
  size_t params_size() const { return static_cast<size_t>(out_) * in_ + out_; }
  /// @brief Weights parameter count
  size_t weights_size() const { return static_cast<size_t>(out_) * in_; }
  /// @brief Bias parameter count
  size_t bias_size() const { return static_cast<size_t>(out_); }

  /// @brief Bind parameter and gradient buffers
  void bind(CudaScalar *params, CudaScalar *grads) {
    params_ptr_ = params;
    grads_ptr_ = grads;
  }

  /// @brief Recommended stddev for weight initialization
  CudaScalar init_stddev() const {
    return activation_scale(act_) * std::sqrt(CudaScalar{1.0f} / static_cast<CudaScalar>(in_));
  }

  /**
   * @brief Forward pass: Z = W*X + b, A = act(Z)
   * @param handle cuBLAS handle
   * @param input Input matrix (in x batch)
   * @param batch Batch size
   * @param output Output matrix (out x batch)
   */
  void forward(CublasHandle &handle, const CudaScalar *input, int batch, CudaScalar *output) {
    const CudaScalar alpha = 1.0f, beta = 0.0f, *W = params_ptr_, *b = params_ptr_ + static_cast<size_t>(out_) * in_;

    cublas_check(
        cublasSgemm(
            handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, out_, batch, in_, &alpha, W, out_, input, in_, &beta, output, out_),
        "cublasSgemm forward");

    launch_add_bias(output, b, out_, batch);
    launch_activation(output, out_ * batch, act_);
  }

  /**
   * @brief Backward pass: compute dW, db, and optionally dX
   * @param handle cuBLAS handle
   * @param input Input activations
   * @param output Output activations
   * @param next_grad Gradient w.r.t. output (out x batch), updated in-place
   * @param batch Batch size
   * @param prev_grad Optional gradient w.r.t. input (in x batch)
   */
  void backward(CublasHandle &handle,
      const CudaScalar *input,
      const CudaScalar *output,
      CudaScalar *next_grad,
      int batch,
      CudaScalar *prev_grad) {
    const CudaScalar alpha = 1.0f, beta = 0.0f, *W = params_ptr_;

    CudaScalar *dW = grads_ptr_, *db = grads_ptr_ + static_cast<size_t>(out_) * in_;

    launch_activation_deriv(next_grad, output, out_ * batch, act_);

    cublas_check(
        cublasSgemm(
            handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, out_, in_, batch, &alpha, next_grad, out_, input, in_, &beta, dW, out_),
        "cublasSgemm dW");

    launch_sum_rows(next_grad, db, out_, batch);

    if (prev_grad) {
      cublas_check(cublasSgemm(handle.get(),
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       in_,
                       batch,
                       out_,
                       &alpha,
                       W,
                       out_,
                       next_grad,
                       out_,
                       &beta,
                       prev_grad,
                       in_),
          "cublasSgemm dX");
    }
  }

private:
  int in_ = 0, out_ = 0;                                    ///< Layer dimensions
  ActivationType act_ = ActivationType::Linear;             ///< Activation type
  CudaScalar *params_ptr_ = nullptr, *grads_ptr_ = nullptr; ///< Bound parameter buffers

public:
  /// @brief Raw parameter pointer for this layer
  const CudaScalar *params_ptr() const { return params_ptr_; }
  /// @brief Raw gradient pointer for this layer
  const CudaScalar *grads_ptr() const { return grads_ptr_; }
};

} // namespace cuda_mlp
