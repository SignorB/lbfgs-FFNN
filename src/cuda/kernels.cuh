#pragma once

#include "common.cuh"
#include "cublas_handle.cuh"
#include <cmath>

namespace cuda_mlp {

inline void device_set_zero(double *ptr, size_t n) { cuda_check(cudaMemset(ptr, 0, n * sizeof(double)), "cudaMemset"); }

inline void device_copy(double *dst, const double *src, size_t n) {
  cuda_check(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy DtoD");
}

inline double device_dot(CublasHandle &handle, const double *x, const double *y, int n) {
  double result = 0.0;
  cublas_check(cublasDdot(handle.get(), n, x, 1, y, 1, &result), "cublasDdot");
  return result;
}

inline double device_nrm2(CublasHandle &handle, const double *x, int n) {
  double result = 0.0;
  cublas_check(cublasDnrm2(handle.get(), n, x, 1, &result), "cublasDnrm2");
  return result;
}

inline void device_axpy(CublasHandle &handle, int n, double alpha, const double *x, double *y) {
  cublas_check(cublasDaxpy(handle.get(), n, &alpha, x, 1, y, 1), "cublasDaxpy");
}

inline void device_scal(CublasHandle &handle, int n, double alpha, double *x) {
  cublas_check(cublasDscal(handle.get(), n, &alpha, x, 1), "cublasDscal");
}

enum class ActivationType : int {
  Linear = 0,
  Tanh = 1,
  ReLU = 2,
  Sigmoid = 3,
};

inline double activation_scale(ActivationType act) {
  switch (act) {
  case ActivationType::ReLU:
    return 1.41421356; // sqrt(2)
  case ActivationType::Linear:
  case ActivationType::Tanh:
  case ActivationType::Sigmoid:
  default:
    return 1.0;
  }
}

__global__ void add_bias_kernel(double *z, const double *b, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    int row = idx % rows;
    z[idx] += b[row];
  }
}

__global__ void activation_kernel(double *a, int n, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double v = a[idx];
    switch (act) {
    case static_cast<int>(ActivationType::Linear):
      a[idx] = v;
      break;
    case static_cast<int>(ActivationType::Tanh):
      a[idx] = tanh(v);
      break;
    case static_cast<int>(ActivationType::ReLU):
      a[idx] = (v > 0.0) ? v : 0.0;
      break;
    case static_cast<int>(ActivationType::Sigmoid):
      a[idx] = 1.0 / (1.0 + exp(-v));
      break;
    default:
      a[idx] = v;
      break;
    }
  }
}

__global__ void activation_deriv_kernel(double *grad, const double *a, int n, int act) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    double v = a[idx];
    double deriv = 1.0;
    switch (act) {
    case static_cast<int>(ActivationType::Linear):
      deriv = 1.0;
      break;
    case static_cast<int>(ActivationType::Tanh):
      deriv = 1.0 - v * v;
      break;
    case static_cast<int>(ActivationType::ReLU):
      deriv = (v > 0.0) ? 1.0 : 0.0;
      break;
    case static_cast<int>(ActivationType::Sigmoid):
      deriv = v * (1.0 - v);
      break;
    default:
      deriv = 1.0;
      break;
    }
    grad[idx] *= deriv;
  }
}

__global__ void diff_kernel(const double *output, const double *target, double *diff, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    diff[idx] = output[idx] - target[idx];
  }
}

__global__ void sum_rows_kernel(const double *mat, double *out, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      sum += mat[row + col * rows];
    }
    out[row] = sum;
  }
}

inline void launch_add_bias(double *z, const double *b, int rows, int cols) {
  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads>>>(z, b, rows, cols);
  cuda_check(cudaGetLastError(), "add_bias_kernel");
}

inline void launch_activation(double *a, int n, ActivationType act) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  activation_kernel<<<blocks, threads>>>(a, n, static_cast<int>(act));
  cuda_check(cudaGetLastError(), "activation_kernel");
}

inline void launch_activation_deriv(double *grad, const double *a, int n, ActivationType act) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  activation_deriv_kernel<<<blocks, threads>>>(grad, a, n, static_cast<int>(act));
  cuda_check(cudaGetLastError(), "activation_deriv_kernel");
}

inline void launch_diff(const double *output, const double *target, double *diff, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  diff_kernel<<<blocks, threads>>>(output, target, diff, n);
  cuda_check(cudaGetLastError(), "diff_kernel");
}

inline void launch_sum_rows(const double *mat, double *out, int rows, int cols) {
  int threads = 256;
  int blocks = (rows + threads - 1) / threads;
  sum_rows_kernel<<<blocks, threads>>>(mat, out, rows, cols);
  cuda_check(cudaGetLastError(), "sum_rows_kernel");
}

} // namespace cuda_mlp
