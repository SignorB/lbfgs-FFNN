#include "../../src/enzyme/pinn_network_cuda.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using Net = pinn_cuda::PINN<pinn_cuda::Dense<2, 32, pinn_cuda::Tanh>,
    pinn_cuda::Dense<32, 32, pinn_cuda::Tanh>,
    pinn_cuda::Dense<32, 1, pinn_cuda::Linear>>;

static void check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    std::exit(1);
  }
}

__global__ void sgd_update(double *params, const double *grads, double lr, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    params[idx] -= lr * grads[idx];
  }
}

int main() {
  std::printf("=== CUDA PINN Burgers + Enzyme (prototype) ===\n");
  std::printf("Params: %d\n", Net::TotalParams);

  Net net;

  std::vector<double> xs;
  std::vector<double> ts;
  std::vector<int> kind;

  auto push = [&](double x, double t, int k) {
    xs.push_back(x);
    ts.push_back(t);
    kind.push_back(k);
  };

  // Allineiamo il dataset a quello usato sul CPU test: circa 3540 punti.
  for (double x = -1.0; x <= 1.0 + 1e-12; x += 0.001)
    push(x, 0.0, pinn_cuda::SampleType::Initial);
  for (double t = 0.0; t <= 1.0 + 1e-12; t += 0.005) {
    push(-1.0, t, pinn_cuda::SampleType::Boundary);
    push(1.0, t, pinn_cuda::SampleType::Boundary);
  }
  for (double x = -1.0; x <= 1.0 + 1e-12; x += 0.03) {
    for (double t = 0.0; t <= 1.0 + 1e-12; t += 0.06)
      push(x, t, pinn_cuda::SampleType::Collocation);
  }

  const int samples = static_cast<int>(xs.size());

  double *d_xs = nullptr;
  double *d_ts = nullptr;
  int *d_kind = nullptr;
  double *d_params = nullptr;
  double *d_grad = nullptr;
  double *d_loss = nullptr;

  check(cudaMalloc(&d_xs, xs.size() * sizeof(double)), "cudaMalloc xs");
  check(cudaMalloc(&d_ts, ts.size() * sizeof(double)), "cudaMalloc ts");
  check(cudaMalloc(&d_kind, kind.size() * sizeof(int)), "cudaMalloc kind");
  check(cudaMalloc(&d_params, Net::TotalParams * sizeof(double)), "cudaMalloc params");
  check(cudaMalloc(&d_grad, Net::TotalParams * sizeof(double)), "cudaMalloc grad");
  check(cudaMalloc(&d_loss, sizeof(double)), "cudaMalloc loss");

  check(cudaMemcpy(d_xs, xs.data(), xs.size() * sizeof(double), cudaMemcpyHostToDevice), "memcpy xs");
  check(cudaMemcpy(d_ts, ts.data(), ts.size() * sizeof(double), cudaMemcpyHostToDevice), "memcpy ts");
  check(cudaMemcpy(d_kind, kind.data(), kind.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy kind");
  check(cudaMemcpy(d_params, net.params, Net::TotalParams * sizeof(double), cudaMemcpyHostToDevice), "memcpy params");
  check(cudaMemset(d_grad, 0, Net::TotalParams * sizeof(double)), "memset grad");

  std::vector<double> grad(Net::TotalParams, 0.0);
  double loss = 0.0;

  const int steps = 50000;
  const double lr = 5e-3;
  dim3 update_grid((Net::TotalParams + 255) / 256);
  dim3 update_block(256);

  for (int it = 0; it < steps; ++it) {
    check(cudaMemset(d_grad, 0, Net::TotalParams * sizeof(double)), "memset grad");
    pinn_cuda::burgers_loss_and_grad<Net><<<1, 1>>>(d_xs, d_ts, d_kind, samples, d_params, d_grad, d_loss);
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaMemcpy(grad.data(), d_grad, Net::TotalParams * sizeof(double), cudaMemcpyDeviceToHost), "copy grad");
    check(cudaMemcpy(&loss, d_loss, sizeof(double), cudaMemcpyDeviceToHost), "copy loss");

    double gnorm = 0.0;
    for (double g : grad)
      gnorm += g * g;
    gnorm = sqrt(gnorm);

    const int preview = (Net::TotalParams < 8) ? Net::TotalParams : 8;
    std::printf("Iter %03d | loss %.6f | grad_norm %.3e\n", it, loss, gnorm);
    std::printf("  preview grads: ");
    for (int i = 0; i < preview; ++i)
      std::printf("%+8.3e ", grad[i]);
    std::printf("\n");

    sgd_update<<<update_grid, update_block>>>(d_params, d_grad, lr, Net::TotalParams);
    check(cudaDeviceSynchronize(), "sgd update");
  }

  cudaFree(d_xs);
  cudaFree(d_ts);
  cudaFree(d_kind);
  cudaFree(d_params);
  cudaFree(d_grad);
  cudaFree(d_loss);

  return 0;
}
