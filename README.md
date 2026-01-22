# L-BFGS and Stochastic L-BFGS for Optimization

Documentation: https://frabazz.github.io/lbfgs-FFNN/

This project implements advanced Quasi-Newton optimization methods, specifically **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) and its stochastic variant **S-LBFGS**, designed for large-scale minimization problems.


## Algorithms

### 1. L-BFGS (Deterministic)

L-BFGS is a Quasi-Newton method that stores a limited history of $m$ updates to approximate the inverse Hessian.
*   **Mechanism**: It stores pairs of vectors $(s_k, y_k)$ where $s_k = w_{k+1} - w_k$ and $y_k = \nabla F(w_{k+1}) - \nabla F(w_k)$.
*   **Two-Loop Recursion**: Efficiently computes the direction $p_k = -H_k^{-1} \nabla F(w_k)$ without forming the matrix $H_k^{-1}$.
*   **Suitability**: Valid for smooth, deterministic problems (e.g., standard optimizations like Rosenbrock function).


### 2. SGD (Stochastic Gradient Descent)

A standard stochastic gradient descent implementation is provided as a baseline for comparison.
*   **Mechanism**: Updates weights using the gradient estimated from a small mini-batch of data samples: $w_{k+1} = w_k - \eta \nabla f_{S_k}(w_k)$.
*   **Suitability**: Simple and widely used for training neural networks and large-scale machine learning models, though it may require careful tuning of the learning rate $\eta$.

### 3. S-LBFGS (Stochastic L-BFGS)

The S-LBFGS implementation follows the algorithm proposed by *Moritz et al. (2016)*. It effectively integrates curvature information into stochastic optimization using a **stable Hessian update** and **variance reduction**.

#### Key Components:

1. **Variance Reduction (SVRG framework)**: To control the noise in the gradient approximation, we use a semi-stochastic gradient:
    * Every $m$ iterations (an epoch), we compute a **full gradient** $\mu = \nabla F(\tilde{w})$ at a reference point $\tilde{w}$.
    * During the inner loop, we update the reference gradient with mini-batch corrections:

$$
v_t = \nabla f_{S_t}(w_t) - \nabla f_{S_t}(\tilde{w}) + \mu
$$

This $v_t$ is an unbiased estimator of $\nabla F(w_t)$ with reduced variance as $w_t \to \tilde{w}$.

2. **Stable Hessian Update**: Standard BFGS updates are unstable with noisy stochastic gradients. S-LBFGS decouples the Hessian update from the step update:
    * Every $L$ iterations, we compute a stable curvature pair $(s, y)$ using a separate mini-batch $b_H$.
    * $s = \bar{w}\_t - \bar{w}\_{t-1}$ (difference of averaged iterates).
    * $y = \nabla^2 F(\bar{w})(\bar{w}\_t - \bar{w}\_{t-1})$ is approximated accurately, often using Hessian-Vector Products (HVP).
3. **Hessian-Vector Products (HVP)**: The curvature vector $y$ is computed using **finite differences** (or automatic differentiation) on a mini-batch:

$$
\nabla^2 F(w) \cdot s \approx \frac{\nabla F(w + \epsilon s) - \nabla F(w - \epsilon s)}{2\epsilon}
$$

This avoids forming the Hessian matrix while capturing the curvature in the direction $s$.

4.  **Parallelization**:
    The code is parallelized through OpenMP. The costly gradient computations and finite-difference HVPs are parallelized across data points.

---

## Implementation Architecture (CPU + CUDA)

The codebase is split into two concrete backends that share the same high-level flow (define a network, compute loss/gradients, and run an optimizer), but use different data structures and kernels:

### CPU Backend (Eigen + OpenMP)

- **Core optimizer API**: `src/minimizer/` hosts shared minimizer utilities and interfaces (full-batch and stochastic base classes, ring buffer).
- **Optimizers**: `src/minimizer/lbfgs.hpp`, `src/minimizer/bfgs.hpp`, `src/minimizer/gd.hpp`, `src/minimizer/s_gd.hpp`, `src/minimizer/s_lbfgs.hpp`, `src/minimizer/newton.hpp`.
- **Network stack**: `src/network.hpp` and `src/layer.hpp` implement a dense MLP with flat parameter storage and Eigen-based forward/backward.
- **Unified training flow**: `src/unified_optimization.hpp`, `src/unified_launcher.hpp`, and `src/network_wrapper.hpp` provide backend configuration, dataset setup, and optimizer strategies.

### CUDA Backend

- **CUDA primitives**: `src/cuda/device_buffer.cuh` wraps device memory, `src/cuda/cublas_handle.cuh` manages cuBLAS, and `src/cuda/kernels.cuh` hosts custom kernels (activation, loss, etc.).
- **Network stack**: `src/cuda/network.cuh` and `src/cuda/layer.cuh` implement a GPU MLP. Parameters and gradients live in contiguous device buffers; per-layer activations/deltas are allocated per batch; GEMMs use cuBLAS.
- **Optimizers**: `src/cuda/minimizer_base.cuh` defines a CUDA optimizer interface. `src/cuda/gd.cuh`, `src/cuda/sgd.cuh`, and `src/cuda/lbfgs.cuh` implement the training steps, with optional history tracking via `src/iteration_recorder.hpp`.
- **Unified training flow**: uses `src/unified_optimization.hpp`, `src/unified_launcher.hpp`, and `src/network_wrapper.hpp` (CUDA specializations compiled under `__CUDACC__`).

## Organization of the Code

The directory structure is as follows:

```text
./amsc
  ├── build/                     # Build artifacts
  ├── src/                       # Source code
  │   ├── minimizer/             # CPU optimizers + base interfaces
  │   ├── cuda/                  # CUDA backend (network + optimizers + kernels)
  │   ├── enzyme/                # Enzyme PINN experiments
  │   ├── network.hpp            # CPU MLP utilities (Eigen)
  │   ├── layer.hpp              # CPU layers/activations
  │   ├── network_wrapper.hpp    # CPU/CUDA wrapper
  │   ├── unified_optimization.hpp
  │   └── unified_launcher.hpp
  ├── tests/
  │   ├── mnist/                 # CPU/GPU MNIST runners
  │   ├── fashion-mnist/         # CPU/GPU Fashion-MNIST runners
  │   ├── burgers/               # PDE/PINN tests
  │   └── pytorch/               # PyTorch reference scripts
  └── ...
```

## Compiling and Running

We use **CMake** for build configuration.

### Prerequisites
- CMake >= 3.18
- A C++20 compiler
- Eigen3 (required)
- OpenMP (optional, enables Eigen multithreading)
- CUDA toolkit + cuBLAS (optional, for GPU targets)

### Build (CPU only)
```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=OFF
make
```

### Build (CPU + CUDA)
```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=ON
make
```
Notes: If CUDA is enabled but no CUDA compiler is found, CUDA targets are skipped.

### Run examples
```bash
./build/test_runner
./build/test_runner_autodiff
./build/test_mnist_cpu
./build/test_fashion_cpu
```

### Run CUDA examples (when built with CUDA)
```bash
./build/test_mnist_gpu
./build/test_fashion_gpu
./build/test_fashion_gpu_deep
```

## PINN Experiments (Burgers)

Standalone CMake is available under `tests/burgers` to keep the main project build unchanged. This target requires Clang and the Enzyme plugin.

Before building the Burgers tests, make sure Enzyme is compiled following the official guide https://enzyme.mit.edu/Installation/ or by building the copy under `lib/Enzyme/enzyme`:
```bash
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
ninja
ninja check-enzyme
```

We tested on `clang++-19`, but this should work with Clang >= 14.

### Build
```bash
cmake -S tests/burgers -B build-burgers \
  -DCMAKE_CXX_COMPILER=clang++-19 \
  -DENZYME_PLUGIN_PATH=/path/to/ClangEnzyme-19.so

cmake --build build-burgers -j
```

### Run
```bash
./build-burgers/test_burgers_parallel
```
