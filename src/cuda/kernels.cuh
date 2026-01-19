#pragma once

#include "common.cuh"
#include "cublas_handle.cuh"
#include <cmath>

namespace cuda_mlp {

/**
 * @brief Set device memory to zero.
 * @param ptr Device pointer.
 * @param n Number of elements.
 */
inline void device_set_zero(CudaScalar *ptr, size_t n) {
  cuda_check(cudaMemset(ptr, 0, n * sizeof(CudaScalar)), "cudaMemset");
}

/**
 * @brief Copy device-to-device.
 * @param dst Destination device pointer.
 * @param src Source device pointer.
 * @param n Number of elements.
 */
inline void device_copy(CudaScalar *dst, const CudaScalar *src, size_t n) {
  cuda_check(cudaMemcpy(dst, src, n * sizeof(CudaScalar), cudaMemcpyDeviceToDevice), "cudaMemcpy DtoD");
}

/// @brief Compute dot product on device using cuBLAS.
inline CudaScalar device_dot(CublasHandle &handle, const CudaScalar *x, const CudaScalar *y, int n) {
  CudaScalar result = 0.0f;
  cublas_check(cublasSdot(handle.get(), n, x, 1, y, 1, &result), "cublasSdot");
  return result;
}

/// @brief Compute Euclidean norm on device using cuBLAS.
inline CudaScalar device_nrm2(CublasHandle &handle, const CudaScalar *x, int n) {
  CudaScalar result = 0.0f;
  cublas_check(cublasSnrm2(handle.get(), n, x, 1, &result), "cublasSnrm2");
  return result;
}

/// @brief y <- alpha * x + y (AXPY) on device using cuBLAS.
inline void device_axpy(CublasHandle &handle, int n, CudaScalar alpha, const CudaScalar *x, CudaScalar *y) {
  cublas_check(cublasSaxpy(handle.get(), n, &alpha, x, 1, y, 1), "cublasSaxpy");
}

/// @brief Scale vector x <- alpha * x on device using cuBLAS.
inline void device_scal(CublasHandle &handle, int n, CudaScalar alpha, CudaScalar *x) {
  cublas_check(cublasSscal(handle.get(), n, &alpha, x, 1), "cublasSscal");
}

/// @brief Supported activation functions.
enum class ActivationType : int {
  Linear = 0,
  Tanh = 1,
  ReLU = 2,
  Sigmoid = 3,
};

/// @brief scaling factor for initialization.
inline CudaScalar activation_scale(ActivationType act) {
  switch (act) {
  case ActivationType::ReLU:
    return 1.41421356f; // sqrt(2)
  case ActivationType::Linear:
  case ActivationType::Tanh:
  case ActivationType::Sigmoid:
  default:
    return 1.0f;
  }
}

/// @brief Kernel: add bias vector to column-major matrix.
__global__ void add_bias_kernel(CudaScalar *z, const CudaScalar *b, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    int row = idx % rows;
    z[idx] += b[row];
  }
}

/// @brief Kernel: apply activation in-place.
__global__ void activation_kernel(CudaScalar *a, int n, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    CudaScalar v = a[idx];
    switch (act) {
    case static_cast<int>(ActivationType::Linear):
      a[idx] = v;
      break;
    case static_cast<int>(ActivationType::Tanh):
      a[idx] = tanhf(v);
      break;
    case static_cast<int>(ActivationType::ReLU):
      a[idx] = (v > 0.0f) ? v : 0.0f;
      break;
    case static_cast<int>(ActivationType::Sigmoid):
      a[idx] = 1.0f / (1.0f + expf(-v));
      break;
    default:
      a[idx] = v;
      break;
    }
  }
}

/// @brief Kernel: multiply gradient by activation derivative.
__global__ void activation_deriv_kernel(CudaScalar *grad, const CudaScalar *a, int n, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    CudaScalar v = a[idx];
    CudaScalar deriv = 1.0f;
    switch (act) {
    case static_cast<int>(ActivationType::Linear):
      deriv = 1.0f;
      break;
    case static_cast<int>(ActivationType::Tanh):
      deriv = 1.0f - v * v;
      break;
    case static_cast<int>(ActivationType::ReLU):
      deriv = (v > 0.0f) ? 1.0f : 0.0f;
      break;
    case static_cast<int>(ActivationType::Sigmoid):
      deriv = v * (1.0f - v);
      break;
    default:
      deriv = 1.0f;
      break;
    }
    grad[idx] *= deriv;
  }
}

/// @brief Kernel: diff = output - target.
__global__ void diff_kernel(const CudaScalar *output, const CudaScalar *target, CudaScalar *diff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    diff[idx] = output[idx] - target[idx];
  }
}

/// @brief Kernel: sum columns (rows x cols) into a row vector.
__global__ void sum_rows_kernel(const CudaScalar *mat, CudaScalar *out, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    CudaScalar sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += mat[row + col * rows];
    }
    out[row] = sum;
  }
}

/// @brief Launch add-bias kernel.
inline void launch_add_bias(CudaScalar *z, const CudaScalar *b, int rows, int cols) {
  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads>>>(z, b, rows, cols);
  cuda_check(cudaGetLastError(), "add_bias_kernel");
}

/// @brief Launch activation kernel.
inline void launch_activation(CudaScalar *a, int n, ActivationType act) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  activation_kernel<<<blocks, threads>>>(a, n, static_cast<int>(act));
  cuda_check(cudaGetLastError(), "activation_kernel");
}

/// @brief Launch activation-derivative kernel.
inline void launch_activation_deriv(CudaScalar *grad, const CudaScalar *a, int n, ActivationType act) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  activation_deriv_kernel<<<blocks, threads>>>(grad, a, n, static_cast<int>(act));
  cuda_check(cudaGetLastError(), "activation_deriv_kernel");
}

/// @brief Launch diff kernel.
inline void launch_diff(const CudaScalar *output, const CudaScalar *target, CudaScalar *diff, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  diff_kernel<<<blocks, threads>>>(output, target, diff, n);
  cuda_check(cudaGetLastError(), "diff_kernel");
}

/// @brief Launch sum-rows kernel.
inline void launch_sum_rows(const CudaScalar *mat, CudaScalar *out, int rows, int cols) {
  int threads = 256;
  int blocks = (rows + threads - 1) / threads;
  sum_rows_kernel<<<blocks, threads>>>(mat, out, rows, cols);
  cuda_check(cudaGetLastError(), "sum_rows_kernel");
}

} // namespace cuda_mlp
