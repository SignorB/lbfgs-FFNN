#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"


#include <eigen3/Eigen/Eigen>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>


#include <random>
#include <numeric>



/**
 * @brief Stochastic-Limited-memory BFGS (S-LBFGS) minimizer.
 * This follows the implementation proposed by Moritz et al. in https://arxiv.org/abs/1508.02087
 *
 * Implements a quasi-Newton optimization method with limited memory, storing
 * only the most recent curvature pairs (s_k, y_k) to approximate the inverse
 * Hessian. Suitable for large-scale unconstrained optimization problems.
 *
 * @tparam V Vector type (e.g., Eigen::VectorXd).
 * @tparam M Matrix type (e.g., Eigen::MatrixXd).
 */
template <typename V, typename M>
class SLBFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
//  using Base::alpha_wolfe;
  using Base::m;

public:

/**
 * @brief Compute the product of the Hessian of f at x with vector v using automatic differentiation.
 *
 * 
 * @param f Function for which to compute the Hessian-vector product.
 * @param x Point at which to evaluate the Hessian.
 * @param v Vector to multiply with the Hessian.
 */
template <typename Func>
Eigen::VectorXd hessian_vector_product(Func &f, const Eigen::VectorXd &x, const Eigen::VectorXd &v) {
    using autodiff::VectorXvar;
    using autodiff::var;


    VectorXvar x_var = x.cast<var>();
    var y = f(x_var); //f(x)
    VectorXvar grad = autodiff::gradient(y, x_var);

    // Compute directional derivative of the gradient in the direction of v

    VectorXvar v_var = v.cast<var>();
    var directional = grad.dot(v_var);

    //hessian*v
    VectorXvar dugrad = autodiff::gradient(directional, x_var);

    return dugrad.cast<double>();
}






  /**
   * @brief Perform the S-LBFGS optimization on the objective function f.
   *
   *
   * @param x Initial guess for the minimizer (passed by value).
   * @param f Objective function to minimize, mapping V → double.
   * @param Gradient Function returning the gradient ∇f(x), mapping V → V.
   * @param m Memory parameter: number of curvature pairs to store.
   * @param M THe number of pair vectors (sj,yj) to use for the Hessian update.
   * @param L The frequency of Hessian updates: every L iterations a new estimate of the Hessian is computed L=10 in the reference paper.
   * @param b Mini-batch size for stochastic gradient estimation.
   * @param b_H Mini-batch size for stochastic Hessian-vector product estimation. 10/20* b is suggested in the reference paper.
   * @param step_size Fixed step size to use in the updates.
   * 
   * 
   * 
   * @param f_components Vector of component functions for finite sum loss function. 
   * @param Gradient_components Vector of gradient functions for finite sum loss function.
   * 
   * @return The final estimate of the minimizer.
   */

V stochastic_solve(V x, std::vector<VecFun<V, double>> &f_components, std::vector<GradFun<V>> &Gradient_components, int m, int M, int L, int b, int b_H, double step_size) {

    int r=0;
    
    //M H = M::Identity(x.size(), x.size()); //initialize H as identity

    std::vector<double> u_list;       //stored iterates for Hessian update
    std::vector<V> s_list;        // displacement u_r − u_r-1
    std::vector<V> y_list;        // Hessian action on displacement
    std::vector<double> rho_list; // Scalars ρ_k = 1 / (y_kᵀ s_k)

    //not sure if this seed should be fixed or random and if it should be called here or inside the loop
    int seed=56;
    std::mt19937 rng(seed); 

    const size_t N = x.size();
    
    while (_iters < _max_iters) { //todo add convergence check 
    // Sample minibatch for gradient estimation
    V full_gradient = V::Zero(x.size());
    for (size_t i=0; i < N; ++i) {
      full_gradient += Gradient_components[i](x);
    }
  
    V xt = x;
    V variance_reduced_gradient = V::Zero(x.size());
    

    for (t=0; t < m ; ++t){

      
      auto minibatch_indices = sample_minibatch_indices(N, b, rng);
      
      V grad_estimate_xt = V::Zero(x.size());
      V grad_estimate_wk = V::Zero(x.size());
      
      for (size_t i=0; i < minibatch_indices.size(); ++i) {
        size_t idx = minibatch_indices[i];
        grad_estimate_xt += Gradient_components[idx](xt);
        grad_estimate_wk += Gradient_components[idx](x);
      }
    
      variance_reduced_gradient = (grad_estimate_xt - grad_estimate_wk) + full_gradient;

      xt = xt - step_size* H * variance_reduced_gradient;


      if (t%L==0 && t>0){
        //hessian update
        r++;
        
        V u = V::Zero(x.size());
        for (int j=t-L; j<t; ++j){
          u += x(j);
        }
        u /= L;
        u_list.push_back(u); //todo Formaggia disse qualcosa su pushback da controllare se va bene
        auto minibatch_indices_H = sample_minibatch_indices(N, b_H, rng);
        V s = u_list[r] - u_list[r - 1]; //displacement
        s_list.push_back(s);

        V y = V::Zero(x.size());
        for (size_t i=0; i < minibatch_indices_H.size(); ++i) {
          size_t idx = minibatch_indices_H[i];
          y += hessian_vector_product(f_components[idx], u_list[r - 1], s);
        }
      y_list.push_back(y);
      }
      



    }
      



    return x;
  }





 private:
  //returns uniform sample of a minibatch
  static   std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng) {
    
    if (N == 0 || batch_size == 0) {
        std::cerr << "Warning: sampling minibatch from empty dataset or with batch_size=0." << std::endl;
      return {};
    }

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) {
      idx[i] = i;
    }


    if (batch_size >= N) {
        std::cerr << "Warning: batch_size >= dataset size. Returning full dataset indices." << std::endl;
        return idx;
        }

    //Fisher Yates shffle
    for (size_t i = 0; i < batch_size; ++i) {
      std::uniform_int_distribution<size_t> dist(i, N -1);
      size_t j =dist(rng);
      std::swap(idx[i], idx[j]);
    }
    idx.resize(batch_size);
    return idx;
    }
};

}
