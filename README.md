# L-BFGS and Stochastic L-BFGS for Optimization

Documentation: https://frabazz.github.io/lbfgs-FFNN/

This project implements advanced Quasi-Newton optimization methods, specifically **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) and its stochastic variant **S-LBFGS**, designed for large-scale and finite-sum minimization problems.

## Problem Statement

We consider the classical unconstrained non-linear optimization problem:

$$
\min_{w \in \mathbb{R}^d} F(w)
$$

where $F : \mathbb{R}^d \to \mathbb{R}$ is a smooth objective function.

### Deterministic Optimization (L-BFGS)

In the standard setting, we assume we can evaluate $F(w)$ and its full gradient $\nabla F(w)$ precisely. A standard second-order method is **Newton’s method**:

$$
w_{k+1} = w_k - H_k^{-1} \nabla F(w_k)
$$

where $H_k = \nabla^2 F(w_k)$ is the Hessian. While Newton's method has quadratic convergence, it is computationally expensive ($O(d^3)$ or $O(d^2)$ per step). **L-BFGS** approximates the product $H_k^{-1} v$ using a history of the last $m$ updates, reducing memory complexity to $O(md)$ and time complexity to $O(md)$.

### Stochastic Optimization (S-LBFGS)

We specifically address the **Finite Sum Minimization** problem, which is central to machine learning and neural network training:

$$
F(w) = \frac{1}{N} \sum_{i=1}^N f_i(w)
$$

where $N$ is the number of data points (or component functions), and $f_i(w)$ calculates the loss for the $i$-th sample (e.g., squared error $f_i(w) = \frac{1}{2}\|h(x_i; w) - y_i\|^2$).

For large $N$, computing the full gradient $\nabla F(w) = \frac{1}{N} \sum \nabla f_i(w)$ at every iteration is prohibitively expensive. **Stochastic Gradient Descent (SGD)** addresses this by using a mini-batch $S \subset \{1, \dots, N\}$:

$$
\nabla F(w) \approx \frac{1}{|S|} \sum_{i \in S} \nabla f_i(w)
$$

However, SGD suffers from high variance and requires diminishing step sizes, leading to slow convergence. **S-LBFGS** combines the curvature information of L-BFGS with variance reduction techniques to achieve faster convergence in stochastic settings.

---

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

1.  **Variance Reduction (SVRG framework)**:
    To control the noise in the gradient approximation, we use a semi-stochastic gradient:
    *   Everything $m$ iterations (an epoch), we compute a **full gradient** $\mu = \nabla F(\tilde{w})$ at a reference point $\tilde{w}$.
    *   During the inner loop, we update the reference gradient with mini-batch corrections:

        $$
        v_t = \nabla f_{S_t}(w_t) - \nabla f_{S_t}(\tilde{w}) + \mu
        $$
    This $v_t$ is an unbiased estimator of $\nabla F(w_t)$ with reduced variance as $w_t \to \tilde{w}$.

2.  **Stable Hessian Update**:
    Standard BFGS updates are unstable with noisy stochastic gradients. S-LBFGS decouples the Hessian update from the step update:
    *   Every $L$ iterations, we compute a stable curvature pair $(s, y)$ using a separate mini-batch $b_H$.
    *   $s = \bar{w}_t - \bar{w}_{t-1}$ (difference of averaged iterates).
    *   $y = \nabla^2 F(\bar{w})(\bar{w}_t - \bar{w}_{t-1})$ is approximated accurately, often using Hessian-Vector Products (HVP).

3.  **Hessian-Vector Products (HVP)**:
    The curvature vector $y$ is computed using **finite differences** (or automatic differentiation) on a mini-batch:

    $$
    \nabla^2 F(w) \cdot s \approx \frac{\nabla F(w + \epsilon s) - \nabla F(w - \epsilon s)}{2\epsilon}
    $$
    This avoids forming the Hessian matrix while capturing the curvature in the direction $s$.

4.  **Parallelization**:
    The code includes an OpenMP-optimized version (`s_lbfgs_parallel.hpp`) that parallelizes the costly gradient computations and finite-difference HVPs across data points.

---

## Implementation Architecture (CPU + CUDA)

The codebase is split into two concrete backends that share the same high-level flow (define a network, compute loss/gradients, and run an optimizer), but use different data structures and kernels:

### CPU Backend (Eigen + OpenMP)

- **Core optimizer API**: `src/minimizer_base.hpp` exposes a common interface plus line search and helpers for automatic differentiation (autodiff and optional Enzyme).
- **Optimizers**: `src/lbfgs.hpp`, `src/bfgs.hpp`, `src/gd.hpp`, `src/s_gd.hpp`, `src/s_lbfgs.hpp`, `src/s_lbfgs_parallel.hpp`, `src/newton.hpp` implement deterministic and stochastic solvers on top of the base class.
- **Network stack**: `src/network.hpp` and `src/layer.hpp` implement a dense MLP, with parameters packed in a flat array and gradients accumulated during backprop. Eigen matrices are used for forward/backward computations.

### CUDA Backend

- **CUDA primitives**: `src/cuda/device_buffer.cuh` wraps device memory, `src/cuda/cublas_handle.cuh` manages cuBLAS, and `src/cuda/kernels.cuh` hosts custom kernels (activation, loss, etc.).
- **Network stack**: `src/cuda/network.cuh` and `src/cuda/layer.cuh` implement a GPU MLP. Parameters and gradients live in contiguous device buffers; per-layer activations/deltas are allocated per batch; GEMMs use cuBLAS.
- **Optimizers**: `src/cuda/minimizer_base.cuh` defines a CUDA optimizer interface. `src/cuda/gd.cuh`, `src/cuda/sgd.cuh`, and `src/cuda/lbfgs.cuh` implement the training steps, with optional history tracking via `src/iteration_recorder.hpp`.
- **Experiment runner**: `tests/cuda/launcher.hpp` wires datasets, networks, and optimizers together, logs CSV summaries, and produces history logs used by the plotting scripts.

## Organization of the Code

The directory structure is as follows:

```text
./amsc
  ├── build/           # Build artifacts
  ├── src/             # Source code
  │   ├── lbfgs.hpp             # Deterministic L-BFGS
  │   ├── s_lbfgs.hpp           # Stochastic L-BFGS implementation (Serial)
  │   ├── s_lbfgs_parallel.hpp  # Stochastic L-BFGS implementation (OpenMP)
  │   ├── network.hpp           # CPU MLP utilities (Eigen)
  │   └── cuda/                 # CUDA backend (network + optimizers + kernels)
  ├── tests/
  │   ├── mnist/                # CPU experiments (Eigen)
  │   └── cuda/                 # CUDA experiments (MNIST/Fashion)
  └── ...
```

## Compiling and Running

We use **CMake** for build configuration.

## EnzymeAD on Newer NVIDIA GPUs (Docker Workflow)

To test EnzymeAD on an NVIDIA GTX5070 GPU, we had to work around driver and CUDA version constraints. CUDA 12.8 is required by the host driver, while Enzyme supports up to CUDA 11.5. For this reason we created a Docker image located at `environment/Dockerfile`. We compile inside the container targeting `sm_80`, and then run the binary on the newer GPU using CUDA forward compatibility.

### Build the Docker image

```bash
docker build -t cuda11-enzyme -f environment/Dockerfile .
```

### Run the Docker container

```bash
docker run -it --rm --privileged --user root -v /:/host:Z cuda11-enzyme /bin/bash
```

### Compile the PINN binary inside the container

```bash
clang++-15 -std=c++17 /host/home/gio/amsc/tests/cuda/pinn_burgers.cu \
  -fplugin=/usr/src/Enzyme/enzyme/build/Enzyme/ClangEnzyme-15.so \
  -O3 --cuda-gpu-arch=sm_80 -lcudart -L/usr/local/cuda-11.5/lib64 -lcublas
```

### Prerequisites
*   C++ compiler with C++17 support (e.g., GCC, Clang)
*   CMake (>= 3.10)
*   Eigen3 (Linear algebra library)
*   OpenMP (Optional, for parallel S-LBFGS)
*   (Optional) CUDA toolkit for GPU support

### Build Instructions

```bash
mkdir build
cd build
cmake .. -DENABLE_CUDA=OFF  # Default is OFF, use -DENABLE_CUDA=ON to enable CUDA
make
```
