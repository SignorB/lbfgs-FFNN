#include "../../src/enzyme/pinn_network.hpp"
#include "../../src/lbfgs.hpp"
#include <iomanip>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense> 

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

Real diff_input(const Real* xt_ptr, const Real* p, int index) {
    Real input_grad[2] = {0.0, 0.0};
    __enzyme_autodiff((void*)MyPINN::forward_static, 
                      enzyme_dup, xt_ptr, input_grad, 
                      enzyme_const, p);
    return input_grad[index];
}

Real calc_ux(const Real* xt_ptr, const Real* p) {
    return diff_input(xt_ptr, p, 0);
}

Real pde_residual(Real x, Real t, const Real* p) {
    Real xt[2] = {x, t};
    Real nu = 0.3 / M_PI; 

    Real u = MyPINN::forward_static(xt, p);
    Real ut = diff_input(xt, p, 1);
    Real ux = diff_input(xt, p, 0);

    Real uxx = 0.0;
    Real grad_ux[2] = {0.0, 0.0};
    __enzyme_autodiff((void*)calc_ux, 
                      enzyme_dup, xt, grad_ux, 
                      enzyme_const, p);
    uxx = grad_ux[0];

    return ut + u * ux - nu * uxx;
}

struct Point { Real x, t; };
struct Data { 
    std::vector<Point> collocation; 
    std::vector<Point> initial;      
    std::vector<Point> boundary;    
};

Real loss_function(Real* p, Data* data) {
   
    
    Real loss_ic = 0.0;
    Real loss_bc = 0.0;
    Real loss_pde = 0.0;
    
    size_t N_ic = data->initial.size();
    size_t N_bc = data->boundary.size();
    size_t N_pde = data->collocation.size();
    
    for (auto& pt : data->initial) {
        Real u = MyPINN::forward_static(&(pt.x), p); 
        Real target = std::sin(M_PI * pt.x);
        loss_ic += (u - target) * (u - target);
    }
    if(N_ic > 0) loss_ic = loss_ic / N_ic;

    for (auto& pt : data->boundary) {
        Real u = MyPINN::forward_static(&(pt.x), p);
        loss_bc += u * u;
    }
    if(N_bc > 0) loss_bc = loss_bc / N_bc;
    
    for (auto& pt : data->collocation) {
        Real r = pde_residual(pt.x, pt.t, p);
        loss_pde += r * r;
    }
    if(N_pde > 0) loss_pde = loss_pde / N_pde;
    
    const Real w_ic = 1.0;
    const Real w_bc = 2.0;
    const Real w_pde = 4.0;
    

    std::cout << " | PDE: " << loss_pde 
              << " | IC: " << loss_ic 
              << " | BC: " << loss_bc 
              << " | N: " << N_pde << std::endl;
    

    return w_bc * loss_bc + w_ic * loss_ic + w_pde * loss_pde;
}

int main() {
    std::cout << "=== Burgers PINN con Enzyme ===\n";
    
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

    for(Real x=-1; x<=X; x+=int_dx)
        for(Real t=0; t<=1; t+=int_dt)
            data.collocation.push_back({x, t});

    auto solver = std::make_shared<LBFGS<VectorXr, MatrixXr>>();
    solver->setHistorySize(128);    
    solver->setMaxLineIters(150);   
    solver->setArmijoMaxIter(80);   

    solver->setMaxIterations(300);

    VectorXr w = Eigen::Map<VectorXr>(net.params.data(), MyPINN::TotalParams);
    
    w = solver->solve_with_enzyme<loss_function>(w, &data);

    std::cout << "outputing file...";
    std::ofstream outFile("burgers_test_extrapolation.csv");
    outFile << "x,t,u,type\n"; 


    for (Real t : {0.0f, 0.5f, 1.0f}) {
      for (Real x = -1.0; x <= 1.0; x += 0.02) {
        Real xt[2] = {x, t};
        Real u = MyPINN::forward_static(xt, w.data());
        outFile << x << "," << t << "," << u << ",0\n";
      }
    }



    for (Real x : {-1.5f, 1.5f}) {
      for (Real t = 0.0; t <= 1.0; t += 0.2) {
        Real xt[2] = {x, t};
        Real u = MyPINN::forward_static(xt, w.data());
        outFile << x << "," << t << "," << u << ",1\n";
      }
    }



    for (Real t = 1.1; t <= 1.5; t += 0.1) {
      for (Real x = -1.0; x <= 1.0; x += 0.05) {
        Real xt[2] = {x, t};
        Real u = MyPINN::forward_static(xt, w.data());
        outFile << x << "," << t << "," << u << ",2\n";
      }
    }

    outFile.close();
    std::cout << "File 'burgers_test_extrapolation.csv' generated.\n";
    
   
    return 0;
}
