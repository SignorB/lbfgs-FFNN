#pragma once

#include "../common.hpp"
#include "../iteration_recorder.hpp"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Eigen>
#include <Eigen/IterativeLinearSolvers>
#include <vector>

extern "C" {
    extern double __enzyme_autodiff(void*, ...);
    extern int enzyme_dup;
    extern int enzyme_const;
    extern int enzyme_out;
}

template <typename V, typename M>
class MinimizerBase {
public:
    virtual ~MinimizerBase() = default;

    /**
     * @brief Returns the number of iterations performed.
     * @return Number of iterations.
     */
    int iterations() const noexcept { return _iters; }

    /**
     * @brief Returns the tolerance for the stopping criterion.
     * @return Tolerance value.
     */
    double tolerance() const noexcept { return _tol; }
    
    /**
     * @brief Sets the maximum number of iterations.
     * @param max_iters Maximum iterations.
     */
    void setMaxIterations(int max_iters) noexcept { _max_iters = max_iters; }

    /**
     * @brief Sets the tolerance for the stopping criterion.
     * @param tol Tolerance value.
     */
    void setTolerance(double tol) noexcept { _tol = tol; }

    /**
     * @brief Sets the initial Hessian approximation.
     * @param b Initial Hessian matrix.
     */
    void setInitialHessian(M b) noexcept { _B = b; }

    /**
     * @brief Sets the Hessian function.
     * @param hessFun Hessian function.
     */
    void setHessian(const HessFun<V, M> &hessFun) noexcept { _hessFun = hessFun; }

    /**
     * @brief Sets the maximum number of iterations for the Armijo line search.
     * @param max_iter Maximum line search iterations.
     */
    void setArmijoMaxIter(double max_iter) noexcept { armijo_max_iter = max_iter; }

    /**
     * @brief Sets the maximum number of iterations for the line search.
     * @param max_iters Maximum line search iterations.
     */
    void setMaxLineIters(double max_iters) noexcept { max_line_iters = max_iters; }

    /**
     * @brief Sets the history size for L-BFGS.
     * @param history_size History size.
     */
    void setHistorySize(size_t history_size) noexcept { m = history_size; }

    /**
     * @brief Attach a recorder for loss/gradient history.
     * @param recorder Recorder to receive iteration data.
     */
    void setRecorder(IterationRecorder<CpuBackend> *recorder) noexcept { recorder_ = recorder; }

    /**
     * @brief Sets the stochastic parameters for optimization.
     * @param stochastic_m Stochastic m.
     * @param M_param Stochastic M parameter.
     * @param L Stochastic L.
     * @param b Stochastic b.
     * @param b_H Stochastic b_H.
     * @param step_size Step size.
     */
    void setStochasticParams(int stochastic_m, int M_param, int L, int b, int b_H, double step_size) noexcept {
        this->stochastic_m = stochastic_m;
        this->M_param = M_param;
        this->L = L;
        this->b = b;
        this->b_H = b_H;
        this->step_size = step_size;
    }

    /**
     * @brief Pure virtual solve function to be implemented by derived classes.
     * @param x Initial guess.
     * @param f Objective function.
     * @param Gradient Gradient function.
     * @return Optimized vector.
     */
    virtual V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) = 0;

    /**
     * @brief Solves the optimization problem using automatic differentiation.
     * @param x Initial guess.
     * @param f_ad Objective function using autodiff types.
     * @return Optimized vector.
     */
    V solve(V x, VecFun<autodiff::VectorXvar, autodiff::var> &f_ad) {
        GradFun<V> gradient_wrapper = [&](V x_double) -> V {
            autodiff::VectorXvar x_var = x_double.template cast<autodiff::var>();
            autodiff::var y = f_ad(x_var);
            Eigen::VectorXd grad = autodiff::gradient(y, x_var);
            return grad;
        };

        VecFun<V, double> f_double = [&](V x_val) {
            autodiff::VectorXvar x_var = x_val.template cast<autodiff::var>();
            return autodiff::val(f_ad(x_var));
        };

        return solve(x, f_double, gradient_wrapper);
    }

    /**
     * @brief Solves the optimization problem using Enzyme for automatic differentiation.
     * @tparam Func Function pointer to the objective function.
     * @tparam UserData Type of user data.
     * @param x Initial guess.
     * @param data Pointer to user data.
     * @param data_is_const Boolean indicating if user data is constant.
     * @return Optimized vector.
     */
    template <auto Func, typename UserData>
    V solve_with_enzyme(V x, UserData* data, bool data_is_const = true) {
        VecFun<V, double> f_wrapper = [&](V val) -> double {
            return Func(val.data(), data);
        };

        GradFun<V> grad_wrapper = [&](V val) -> V {
            V grad(val.size());
            grad.setZero();

            if (data_is_const) {
                __enzyme_autodiff((void*)Func, 
                                  enzyme_dup, val.data(), grad.data(),
                                  enzyme_const, data);
            } else {
                __enzyme_autodiff((void*)Func, 
                                  enzyme_dup, val.data(), grad.data(),
                                  enzyme_dup, data, (void*)0); 
            }
            return grad;
        };

        return solve(x, f_wrapper, grad_wrapper);
    }

protected:
    unsigned int _max_iters = 1000;
    unsigned int _iters = 0;
    double _tol = 1.e-10;
    M _B;
    HessFun<V, M> _hessFun;

    double armijo_max_iter = 20;
    double max_line_iters = 50;
    size_t m = 16;
    double alpha_wolfe = 1e-3;
    double c1 = 1e-4;
    double c2 = 0.9;
    double rho = 0.5;

    int stochastic_m = 10;
    int M_param = 10;
    int L = 10;
    int b = 10;
    int b_H = 16;
    double step_size = 0.01;
    IterationRecorder<CpuBackend> *recorder_ = nullptr;

    /**
     * @brief Performs line search using Wolfe conditions.
     * @param x Current point.
     * @param p Search direction.
     * @param f Objective function.
     * @param Gradient Gradient function.
     * @return Step size alpha.
     */
    double line_search(V x, V p, VecFun<V, double> &f, GradFun<V> &Gradient) {
        double f_old = f(x);
        double grad_f_old = Gradient(x).dot(p);
        double inf = std::numeric_limits<double>::infinity();
        double alpha_min = 0.0;
        double alpha_max = inf;
        double alpha = 1.0;

        for (int i = 0; i < max_line_iters; ++i) {
            V x_new = x + alpha * p;
            double f_new = f(x_new);

            if (f_new > f_old + c1 * alpha * grad_f_old) {
                alpha_max = alpha;
                alpha = rho * (alpha_min + alpha_max);
                continue;
            }

            double grad_f_new_dot_p = Gradient(x_new).dot(p);

            if (grad_f_new_dot_p < c2 * grad_f_old) {
                alpha_min = alpha;
                if (alpha_max == inf)
                    alpha *= 2;
                else
                    alpha = rho * (alpha_min + alpha_max);
                continue;
            }
            return alpha;
        }
        return alpha;
    }
};
