#include "../../src/enzyme/pinn_network.hpp"
#include "../../src/minimizer/lbfgs.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

using namespace cpu_mlp;

using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

extern "C" {
extern Real __enzyme_autodiff(void *, ...);
extern Real __enzyme_fwddiff(void *, ...);
int enzyme_dup;
int enzyme_const;
int enzyme_out;
}

constexpr unsigned int net_size = 20;
using MyPINN = PINN<Dense<2, net_size, Tanh>,
    Dense<net_size, net_size, Tanh>,
    Dense<net_size, net_size, Tanh>,
    Dense<net_size, 1, Linear>>;

struct Point {
  Real x, t;
  Real target;
};
struct Data {
  std::vector<Point> collocation;
  std::vector<Point> initial;
  std::vector<Point> boundary;
};

static Real get_u(const Real *xt, const Real *p) { return MyPINN::forward_static(xt, p); }

static Real calc_du(const Real *xt, const Real *p, int index) {
  Real d_xt[2] = {0.0, 0.0};
  d_xt[index] = 1.0;
  return __enzyme_fwddiff((void *)get_u, enzyme_dup, xt, d_xt, enzyme_const, p);
}

static Real get_ux(const Real *xt, const Real *p) { return calc_du(xt, p, 0); }

static Real pde_residual(Real x, Real t, const Real *p) {
  Real xt[2] = {x, t};
  Real nu = 0.01 / M_PI;

  Real u = get_u(xt, p);
  Real ut = calc_du(xt, p, 1);
  Real ux = calc_du(xt, p, 0);

  Real d_xt_xx[2] = {1.0, 0.0};
  Real uxx = __enzyme_fwddiff((void *)get_ux, enzyme_dup, xt, d_xt_xx, enzyme_const, p);

  return ut + u * ux - nu * uxx;
}

static Real loss_ic_single(const Real *p, const Real *xt_val, Real target) {
  Real u = get_u(xt_val, p);
  Real diff = u - target;
  return diff * diff;
}

static Real loss_bc_single(const Real *p, const Real *xt_val) {
  Real u = get_u(xt_val, p);
  return u * u;
}

static Real loss_pde_single(const Real *p, const Real *xt_val) {
  Real r = pde_residual(xt_val[0], xt_val[1], p);
  return r * r;
}

static Real weighted_loss_ic(const Real *p, const Real *xt, Real target, Real weight) {
  return weight * loss_ic_single(p, xt, target);
}

static Real weighted_loss_bc(const Real *p, const Real *xt, Real weight) { return weight * loss_bc_single(p, xt); }

static Real weighted_loss_pde(const Real *p, const Real *xt, Real weight) { return weight * loss_pde_single(p, xt); }

int main() {
  std::cout << "=== Burgers PINN Parallel (Manual Gradient Accumulation) ===\n";
  std::cout << "Threads: " << omp_get_max_threads() << "\n";

  MyPINN net;
  Data data;

  Real dx = 0.001;
  Real dt = 0.005;
  Real int_dx = 0.01;
  Real int_dt = 0.02;
  Real X = 1.0;
  Real T = 1.0;

  for (Real x = -1; x <= X; x += dx) {
    Real target = std::sin(M_PI * x);
    data.initial.push_back({x, 0.0, target});
  }

  for (Real t = 0; t <= T; t += dt) {
    data.boundary.push_back({-1.0, t, 0.0});
    data.boundary.push_back({1.0, t, 0.0});
  }

  for (Real x = -1; x <= X; x += int_dx)
    for (Real t = 0; t <= 1; t += int_dt)
      data.collocation.push_back({x, t, 0.0});

  std::cout << "PDE Points: " << data.collocation.size() << "\n";

  auto solver = std::make_shared<LBFGS<VectorXr, MatrixXr>>();
  solver->setHistorySize(100);
  solver->setMaxLineIters(100);
  solver->setArmijoMaxIter(50);
  solver->setMaxIterations(5000);

  VectorXr w = Eigen::Map<VectorXr>(net.params.data(), MyPINN::TotalParams);

  const Real w_ic = 20.0;
  const Real w_bc = 20.0;
  const Real w_pde = 1.0;

  VecFun<VectorXr, double> f_loss = [&](const VectorXr &params) -> double {
    const Real *p_ptr = params.data();
    double total_ic = 0.0, total_bc = 0.0, total_pde = 0.0;

#pragma omp parallel reduction(+ : total_ic, total_bc, total_pde)
    {
#pragma omp for nowait
      for (size_t i = 0; i < data.initial.size(); ++i) {
        Real xt[2] = {data.initial[i].x, data.initial[i].t};
        total_ic += loss_ic_single(p_ptr, xt, data.initial[i].target);
      }

#pragma omp for nowait
      for (size_t i = 0; i < data.boundary.size(); ++i) {
        Real xt[2] = {data.boundary[i].x, data.boundary[i].t};
        total_bc += loss_bc_single(p_ptr, xt);
      }

#pragma omp for
      for (size_t i = 0; i < data.collocation.size(); ++i) {
        Real xt[2] = {data.collocation[i].x, data.collocation[i].t};
        total_pde += loss_pde_single(p_ptr, xt);
      }
    }

    if (!data.initial.empty()) total_ic /= data.initial.size();
    if (!data.boundary.empty()) total_bc /= data.boundary.size();
    if (!data.collocation.empty()) total_pde /= data.collocation.size();

    return w_ic * total_ic + w_bc * total_bc + w_pde * total_pde;
  };

  GradFun<VectorXr> grad_loss_real = [&](const VectorXr &params) -> VectorXr {
    VectorXr global_grad = VectorXr::Zero(params.size());
    const Real *p_ptr = params.data();

    Real s_ic = w_ic / std::max(1.0, (double)data.initial.size());
    Real s_bc = w_bc / std::max(1.0, (double)data.boundary.size());
    Real s_pde = w_pde / std::max(1.0, (double)data.collocation.size());

#pragma omp parallel
    {
      VectorXr local_grad = VectorXr::Zero(params.size());

#pragma omp for nowait
      for (size_t i = 0; i < data.initial.size(); ++i) {
        Real xt[2] = {data.initial[i].x, data.initial[i].t};
        __enzyme_autodiff((void *)weighted_loss_ic,
            enzyme_dup,
            p_ptr,
            local_grad.data(),
            enzyme_const,
            xt,
            enzyme_const,
            data.initial[i].target,
            enzyme_const,
            s_ic);
      }

#pragma omp for nowait
      for (size_t i = 0; i < data.boundary.size(); ++i) {
        Real xt[2] = {data.boundary[i].x, data.boundary[i].t};
        __enzyme_autodiff(
            (void *)weighted_loss_bc, enzyme_dup, p_ptr, local_grad.data(), enzyme_const, xt, enzyme_const, s_bc);
      }

#pragma omp for
      for (size_t i = 0; i < data.collocation.size(); ++i) {
        Real xt[2] = {data.collocation[i].x, data.collocation[i].t};
        __enzyme_autodiff(
            (void *)weighted_loss_pde, enzyme_dup, p_ptr, local_grad.data(), enzyme_const, xt, enzyme_const, s_pde);
      }

#pragma omp critical
      {
        global_grad += local_grad;
      }
    }
    return global_grad;
  };
  auto start = std::chrono::high_resolution_clock::now();
  w = solver->solve(w, f_loss, grad_loss_real);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Training Time: " << elapsed.count() << "s\n";

  std::ofstream outFile("burgers_test_extrapolation.csv");
  outFile << "x,t,u,type\n";
  auto write_grid = [&](Real t, int type) {
    for (Real x = -1.0; x <= 1.0; x += 0.02) {
      Real xt[2] = {x, t};
      Real u = get_u(xt, w.data());
      outFile << x << "," << t << "," << u << "," << type << "\n";
    }
  };
  write_grid(0.0, 0);
  write_grid(0.5, 0);
  write_grid(1.0, 0);
  write_grid(1.5, 2);
  outFile.close();

  return 0;
}