
#include "../../src/network_pinn.hpp"
#include "../../src/lbfgs.hpp"
#include "../../src/common.hpp"
#include "../../src/s_lbfgs.hpp"


using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

using var= autodiff::var;
using VectorXvar = Eigen::Matrix<var, Eigen::Dynamic, 1>;
    



double rand_val(double a, double b);


int main() {
  checkParallelism();
  
    int n_internal = 1000;
    int n_bound_spatial = 100;
    int n_bound_time = 100;

    
 
    Network<var> network;
    network.addLayer<2, 16, Tanh>();
    network.addLayer<16, 16, Tanh>();
    network.addLayer<16, 16, Tanh>();
    network.addLayer<16, 1, Linear>();
    network.bindParams();

    Mat data_int = Mat::Zero(n_internal, 2);
    Mat data_bound = Mat::Zero(n_bound_spatial, 2);
    Mat data_time = Mat::Zero(n_bound_time, 2);

    for (int i = 0; i < n_internal; ++i) {
        data_int(i, 0) = rand_val(-1.0, 1.0); // x in [-1,1]
        data_int(i, 1) = rand_val(0.0, 1.0); // t in [0,1]
    }   

    for (int i =0 ; i<n_bound_spatial; ++i){
        data_bound(i,1) = rand_val(0.0, 1.0); // t in [0,1]
    }

    for (int i =0 ; i<n_bound_time; ++i){
        data_time(i,0) = rand_val(-1.0, 1.0); // x in [-1,1]
    }

    auto loss_function = [&] (const VectorXvar &weights)->var{
        var loss_total=0.0;
        network.setParams(weights);
        for (int i = 0; i < n_internal; ++i) {
            var x = data_int(i,0);
            var t = data_int(i,1);
            Eigen::Matrix<var, 2, 1> input;
            input(0) = x;
            input(1) = t;
            var u = network.forward(input)(0,0);

            var u_x =autodiff::derivative(u, wrt(x));
            var u_t =autodiff::derivative(u, wrt(t));
            var u_xx= autodiff::derivative(u_x, wrt(x));

            var f = u_t - 0.01 / M_PI * u_xx + u * u_x;

            loss_total += f * f;}
            
        for (int i=0;i<n_bound_spatial;++i){
            var t= data_bound(i,1);
            Eigen::Matrix<var, 2, 1> input_left;
            input_left(0) = -1.0;
            input_left(1) = t;
            var u_left = network.forward(input_left)(0,0);
            Eigen::Matrix<var, 2, 1> input_right;
            input_right(0) = 1.0;
            input_right(1) = t;
            var u_right = network.forward(input_right)(0,0);
            loss_total += (u_left * u_left + u_right * u_right);
        }

        for (int i=0;i<n_bound_time;++i){
            var x= data_time(i,0);
            Eigen::Matrix<var, 2, 1> input0;
            input0(0) = x;
            input0(1) = 0.0;
            var u0 = network.forward(input0)(0,0);
            var u0_exact = -sin(M_PI * x);
            loss_total += (u0 - u0_exact) * (u0 - u0_exact);
        }

        return loss_total;
    };
    

std::shared_ptr<MinimizerBase<Vec, Mat>> solver = std::make_shared<LBFGS<Vec, Mat>>();
solver->setMaxIterations(500);
solver->setTolerance(1.e-4);

    VectorXvar params_var = network.getParams();

    Vec initial_params(params_var.size());
    for (int i = 0; i < params_var.size(); ++i) {
        initial_params(i) = static_cast<double>(params_var(i));
    }

    //function wrapper necessary to match the expected signature

    std::function<var(VectorXvar)> loss_func_wrapper = loss_function;

    Vec optimized_params = solver->solve(initial_params, loss_func_wrapper);
    

    network.setParams(optimized_params.cast<var>());
    std::cout << "Training ended." << std::endl;

    std::ofstream file("burgers_solution.csv");
    file << "t,x,u\n";

    for(double t=0; t<=1.0; t+=0.02) {
        for(double x=-1.0; x<=1.0; x+=0.02) {
            Eigen::Matrix<var, 2, 1> in; in << x, t;
            double u = static_cast<double>(network.forward(in)(0, 0)); 
            file << t << "," << x << "," << u << "\n";
        }
    }
    file.close();
    std::cout << "Dati salvati in burgers_solution.csv" << std::endl;

    return 0;
}


double rand_val(double a, double b){
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}