// PARALLEL (NOT OPTIMAL) VERSION, 1.5X on 4 cores  

#include "../../src/enzyme/pinn_network.hpp"
#include "../../src/minimizer/lbfgs.hpp"
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <omp.h> 

using VectorXr = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using MatrixXr = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

extern "C" {
    extern Real __enzyme_autodiff(void*, ...);
    int enzyme_dup;
    int enzyme_const;
    int enzyme_out;
}

constexpr unsigned int net_size = 100;
using MyPINN = PINN<
    Dense<2, net_size, Tanh>,
    Dense<net_size, net_size, Tanh>,
    Dense<net_size, net_size, Tanh>,
    Dense<net_size, 1, Linear> 
>;

struct Point { Real x, t; };
struct Data { 
    std::vector<Point> collocation; 
    std::vector<Point> initial;       
    std::vector<Point> boundary;    
};

Real diff_input(const Real* xt_ptr, const Real* p, int index) {
    Real input_grad[2];
    input_grad[0] = 0.0;
    input_grad[1] = 0.0;
    
    __enzyme_autodiff((void*)MyPINN::forward_static, 
                      enzyme_dup, xt_ptr, input_grad, 
                      enzyme_const, p);
    return input_grad[index];
}


Real calc_ux(const Real* xt_ptr, const Real* p) {
    return diff_input(xt_ptr, p, 0);
}

Real pde_residual(Real x, Real t, const Real* p) {

    Real xt[2]; 
    xt[0] = x; 
    xt[1] = t;

    Real nu = 0.3 / M_PI; 

    Real u = MyPINN::forward_static(xt, p);
    Real ut = diff_input(xt, p, 1);
    Real ux = diff_input(xt, p, 0);


    Real uxx = 0.0;
    Real grad_ux[2];
    grad_ux[0] = 0.0; 
    grad_ux[1] = 0.0;

    __enzyme_autodiff((void*)calc_ux, 
                      enzyme_dup, xt, grad_ux, 
                      enzyme_const, p);
    uxx = grad_ux[0];

    return ut + u * ux - nu * uxx;
}

Real loss_function(Real* p, Data* data) {
    Real loss_ic_sum = 0.0;
    Real loss_bc_sum = 0.0;
    Real loss_pde_sum = 0.0;
    

    int N_ic = (int)data->initial.size();
    int N_bc = (int)data->boundary.size();
    int N_pde = (int)data->collocation.size();


    #pragma omp parallel for
    for (int i = 0; i < N_ic; ++i) {
        Real x = data->initial[i].x;
        Real xt[2]; xt[0] = x; xt[1] = 0.0;

        Real u = MyPINN::forward_static(xt, p); 
        Real target = std::sin(M_PI * x);
        Real diff = u - target;
        Real sq_diff = diff * diff;

        #pragma omp atomic
        loss_ic_sum += sq_diff;
    }
    Real loss_ic = (N_ic > 0) ? (loss_ic_sum / N_ic) : 0.0;


    #pragma omp parallel for
    for (int i = 0; i < N_bc; ++i) {
        Real x = data->boundary[i].x;
        Real t = data->boundary[i].t;
        Real xt[2]; xt[0] = x; xt[1] = t;

        Real u = MyPINN::forward_static(xt, p);
        Real sq_u = u * u;

        #pragma omp atomic
        loss_bc_sum += sq_u;
    }
    Real loss_bc = (N_bc > 0) ? (loss_bc_sum / N_bc) : 0.0;
    

    #pragma omp parallel for
    for (int i = 0; i < N_pde; ++i) {
        Real x = data->collocation[i].x;
        Real t = data->collocation[i].t;
        
        Real r = pde_residual(x, t, p);
        Real sq_r = r * r;

        #pragma omp atomic
        loss_pde_sum += sq_r;
    }
    Real loss_pde = (N_pde > 0) ? (loss_pde_sum / N_pde) : 0.0;
    

    const Real w_ic = 1.0;
    const Real w_bc = 2.0;
    const Real w_pde = 4.0;

    return w_bc * loss_bc + w_ic * loss_ic + w_pde * loss_pde;
}

int main() {
    std::cout << "=== Burgers PINN con Enzyme & OpenMP Atomic ===\n";
    MyPINN net;
    Data data;

    Real dx = 0.001;
    Real dt = 0.005;
    Real int_dx = 0.03;
    Real int_dt = 0.06;
    Real X = 1.0;
    Real T = 1.0;
    
    for(Real x=-1; x<=X; x+=dx) data.initial.push_back({x, 0.0});
    
    for(Real t=0; t<=T; t+=dt) {
        data.boundary.push_back({-1.0, t});
        data.boundary.push_back({1.0, t});
    }

    data.collocation.reserve(2000); 
    for(Real x=-1; x<=X; x+=int_dx)
        for(Real t=0; t<=1; t+=int_dt)
            data.collocation.push_back({x, t});

    auto solver = std::make_shared<LBFGS<VectorXr, MatrixXr>>();

    solver->setHistorySize(128);    
    solver->setMaxLineIters(150);   
    solver->setArmijoMaxIter(80);   

    solver->setMaxIterations(6);

    
    VectorXr w = Eigen::Map<VectorXr>(net.params.data(), MyPINN::TotalParams);
    

    w = solver->solve_with_enzyme<loss_function>(w, &data);

    std::cout << "outputing file...\n";
    std::ofstream outFile("burgers_test_extrapolation.csv");
    outFile << "x,t,u,type\n"; 
    
    auto write_points = [&](Real t_start, Real t_end, Real t_step, 
                            Real x_start, Real x_end, Real x_step, int type) {
        for (Real t = t_start; t <= t_end; t += t_step) {
            for (Real x = x_start; x <= x_end; x += x_step) {
                Real xt[2]; xt[0] = x; xt[1] = t;
                Real u = MyPINN::forward_static(xt, w.data());
                outFile << x << "," << t << "," << u << "," << type << "\n";
            }
        }
    };

    for (Real t : {0.0f, 0.5f, 1.0f}) {
        for (Real x = -1.0; x <= 1.0; x += 0.02) {
             Real xt[2]; xt[0] = x; xt[1] = t;
             Real u = MyPINN::forward_static(xt, w.data());
             outFile << x << "," << t << "," << u << ",0\n";
        }
    }
    write_points(0.0, 1.0, 0.2, -1.5, -1.5, 1.0, 1);
    write_points(0.0, 1.0, 0.2, 1.5, 1.5, 1.0, 1);
    write_points(1.1, 1.5, 0.1, -1.0, 1.0, 0.05, 2);

    outFile.close();
    
    return 0;
}
