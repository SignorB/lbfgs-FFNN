#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

namespace cuda_mlp {

/// @brief Scalar type used across CUDA kernels and optimizers
using CudaScalar = float;

/**
 * @brief Check a CUDA API call and abort with a message on failure
 * @param err CUDA error code returned by the runtime
 * @param msg Context string describing the operation
 */
inline void cuda_check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
    std::abort();
  }
}

} // namespace cuda_mlp
