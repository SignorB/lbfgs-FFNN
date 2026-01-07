#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

namespace cuda_mlp {

using CudaScalar = float;

inline void cuda_check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
    std::abort();
  }
}

} // namespace cuda_mlp
