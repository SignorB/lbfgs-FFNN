#pragma once

#include "../src/cuda/device_buffer.cuh"
#include "../src/cuda/network.cuh"
#include <Eigen/Core>
#include <iostream>
#include <vector>

namespace cuda_tests {

template <typename Scalar>
struct ReportStats {
  long correct = 0;
  long total = 0;
  Scalar mse = static_cast<Scalar>(0);
  Scalar accuracy = static_cast<Scalar>(0);
};

template <typename Scalar, typename Mat>
inline ReportStats<Scalar> compute_report(cuda_mlp::CudaNetwork &network,
                                          const cuda_mlp::DeviceBuffer<Scalar> &inputs,
                                          const Mat &targets,
                                          int batch) {
  network.forward_only(inputs.data(), batch);

  const int out_size = network.output_size();
  const std::size_t output_count = static_cast<std::size_t>(out_size) * static_cast<std::size_t>(batch);
  std::vector<Scalar> output(output_count);
  if (!output.empty()) {
    network.copy_output_to_host(output.data(), output.size());
  }

  ReportStats<Scalar> stats;
  stats.total = batch;

  if (output_count == 0 || batch <= 0) {
    return stats;
  }

  const Scalar *target_ptr = targets.data();
  Scalar mse = static_cast<Scalar>(0);
  long correct = 0;

  for (int i = 0; i < batch; ++i) {
    int pred_idx = 0;
    int true_idx = 0;
    Scalar pred_max = output[static_cast<std::size_t>(i) * out_size];
    Scalar true_max = target_ptr[static_cast<std::size_t>(i) * out_size];
    Scalar diff0 = pred_max - true_max;
    mse += diff0 * diff0;

    for (int r = 1; r < out_size; ++r) {
      Scalar val = output[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * out_size];
      if (val > pred_max) {
        pred_max = val;
        pred_idx = r;
      }
      Scalar tval = target_ptr[static_cast<std::size_t>(r) + static_cast<std::size_t>(i) * out_size];
      if (tval > true_max) {
        true_max = tval;
        true_idx = r;
      }
      Scalar diff = val - tval;
      mse += diff * diff;
    }
    if (pred_idx == true_idx) {
      ++correct;
    }
  }

  stats.correct = correct;
  stats.mse = mse / static_cast<Scalar>(output_count);
  stats.accuracy = (static_cast<Scalar>(correct) / static_cast<Scalar>(stats.total)) * static_cast<Scalar>(100);
  return stats;
}

template <typename Scalar, typename Mat>
inline void print_report(const char *title,
                         cuda_mlp::CudaNetwork &network,
                         const cuda_mlp::DeviceBuffer<Scalar> &inputs,
                         const Mat &targets,
                         int batch) {
  ReportStats<Scalar> stats = compute_report(network, inputs, targets, batch);

  std::cout << "\n" << title << std::endl;
  std::cout << "Samples: " << stats.total << std::endl;
  std::cout << "Accuracy: " << stats.accuracy << "% (" << stats.correct << "/" << stats.total << ")"
            << std::endl;
  std::cout << "MSE: " << stats.mse << std::endl;
}

} // namespace cuda_tests
