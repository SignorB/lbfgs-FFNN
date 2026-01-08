#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

#include "../src/lbfgs.hpp" 

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct ProblemData {
    int n;
    double param_A; 
    double param_K; 
};

double rosenbrock_enzyme(double* x, ProblemData* data) {
    double val = 0.0;
    int n = data->n;
    double k = data->param_K;

    for (int i = 0; i < n - 1; ++i) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        val += k * t1 * t1 + t2 * t2;
    }
    return val;
}

double rastrigin_enzyme(double* x, ProblemData* data) {
    double val = 0.0;
    int n = data->n;
    double A = data->param_A;
    
    for (int i = 0; i < n; ++i) {
        val += (x[i] * x[i]) - (A * cos(2.0 * M_PI * x[i]));
    }
    return A * n + val;
}

void print_result(const std::string& name, 
                  const Vec& result, 
                  const Vec& expected, 
                  double final_loss,
                  int iters,
                  double time_ms) {
    
    double dist = (result - expected).norm();
    std::cout << "==================================================" << std::endl;
    std::cout << "TEST: " << name << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "  Iterations: " << iters << std::endl;
    std::cout << "  Time:       " << time_ms << " ms" << std::endl;
    std::cout << "  Final Loss: " << final_loss << std::endl;
    std::cout << "  Dist to Min:" << dist << std::endl;
    
    if (dist < 1e-3) {
        std::cout << "  STATUS:     [ SUCCESS ] (Converged to global min)" << std::endl;
    } else {
        std::cout << "  STATUS:     [ WARNING ] (Did not hit exact global min)" << std::endl;
    }
    std::cout << "==================================================\n" << std::endl;
}

int main() {
    std::cout << "Enzyme + L-BFGS Test Suite\n" << std::endl;

    auto solver = std::make_shared<LBFGS<Vec, Mat>>();
    solver->setMaxIterations(2000);
    solver->setTolerance(1e-6);

    {
        int n = 10;
        ProblemData data{n, 0.0, 100.0};
        
        Vec x(n);
        for(int i=0; i<n; ++i) x[i] = (i % 2 == 0) ? -1.2 : 1.0;
        Vec expected = Vec::Ones(n);

        auto start_time = std::chrono::high_resolution_clock::now();
        
        Vec result = solver->solve_with_enzyme<rosenbrock_enzyme>(x, &data);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        double final_loss = rosenbrock_enzyme(result.data(), &data);

        print_result("Rosenbrock (N=10)", result, expected, final_loss, solver->iterations(), elapsed.count());
    }

    {
        int n = 10;
        ProblemData data{n, 10.0, 0.0}; 

        Vec x(n);
        for(int i=0; i<n; ++i) x[i] = 4.5; 
        Vec expected = Vec::Zero(n);

        solver->setMaxIterations(3000);

        auto start_time = std::chrono::high_resolution_clock::now();

        Vec result = solver->solve_with_enzyme<rastrigin_enzyme>(x, &data);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

        double final_loss = rastrigin_enzyme(result.data(), &data);

        print_result("Rastrigin (N=10)", result, expected, final_loss, solver->iterations(), elapsed.count());
    }

    return 0;
}
