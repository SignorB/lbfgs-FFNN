#include "../src/enzyme/pinn_network.hpp"
#include "../src/lbfgs.hpp" 
#include <iomanip>


//stuff needed for enzyme, see docs
extern "C" {
    extern double __enzyme_autodiff(void*, ...);
    int enzyme_dup;
    int enzyme_const;
    int enzyme_out;
}

using MyPINN = PINN<
    Dense<1, 20, Tanh>,
    Dense<20, 20, Tanh>,
    Dense<20, 1, Linear> 
>;



double calc_dudx(const double* x_ptr, const double* p) {
    double du_dx = 0.0;
    __enzyme_autodiff((void*)MyPINN::forward_static, 
                      enzyme_dup, x_ptr, &du_dx, 
                      enzyme_const, p);
    return du_dx;
}

double pde_residual(double x, const double* p) {
   
    double u = MyPINN::forward_static(&x, p);

    double d2u_dx2 = 0.0;
    __enzyme_autodiff((void*)calc_dudx, 
                      enzyme_dup, &x, &d2u_dx2, 
                      enzyme_const, p);
    return d2u_dx2 + u;
}

struct Data { std::vector<double> points; };

double loss_function(double* p, Data* data) {
    double loss = 0.0;
    
    // BC: u(0) = 0
    double zero = 0.0;
    double u0 = MyPINN::forward_static(&zero, p);
    loss += u0 * u0;

    // BC: u'(0) = 1
    double du0 = calc_dudx(&zero, p);
    loss += (du0 - 1.0) * (du0 - 1.0);

    // Physics
    for (double x : data->points) {
        double r = pde_residual(x, p);
        loss += r * r;
    }
    
    return loss;
}

int main() {
    std::cout << "=== Templated PINN with Enzyme ===\n";
    std::cout << "Total Params: " << MyPINN::TotalParams << "\n";

    MyPINN net; 
    Data data;
    for(double x=0; x<=6.28; x+=0.1) data.points.push_back(x);

    auto solver = std::make_shared<LBFGS<Eigen::VectorXd, Eigen::MatrixXd>>();
    solver->setMaxIterations(2000);
    solver->setTolerance(1e-6);

    Eigen::VectorXd w = Eigen::Map<Eigen::VectorXd>(net.params.data(), MyPINN::TotalParams);
    
    w = solver->solve_with_enzyme<loss_function>(w, &data);

    std::cout << "\nValidation:\n";
    double max_err = 0.0;
    for(double x=0; x<=3.14; x+=0.5) {
        double pred = MyPINN::forward_static(&x, w.data());
        double exact = std::sin(x);
        std::cout << "x=" << x << " Pred=" << pred << " Exact=" << exact << "\n";
        max_err = std::max(max_err, std::abs(pred-exact));
    }
    std::cout << "Max Error: " << max_err << std::endl;

    return 0;
}

