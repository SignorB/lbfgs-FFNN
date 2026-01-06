#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"


#include <Eigen/Eigen>
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
template <typename V, typename M,typename D>
class SLBFGS : public MinimizerBase<V, M> {
  using Base = MinimizerBase<V, M>;
  using Base::_iters;
  using Base::_max_iters;
  using Base::_tol;
//  using Base::alpha_wolfe;
  using Base::m;



using S_VecFun=std::function<double(const V&,const D& )>; //loss function must take weights and data as input and returns a double
using S_GradFun=std::function<void(const V&,const D&, V& grad )>; //gradient function must take weights and data as input and returns a vector


public:
  V stochastic_solve(std::vector<V> x, V weights,const S_VecFun &f,const S_GradFun &S_GradFun, int m, int M_param, int L, int b, int b_H, double step_size, int N, bool verbose=false, int print_every=50);
  static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override;

};

  //returns uniform sample of a minibatch
  /**
  * @brief Sample a minibatch of indices from 0 to N-1.
  *
  * @param N Total number of data points.
  * @param batch_size Size of the minibatch to sample.
  * @param rng Random number generator.
  */

template <typename V, typename M, typename D>
std::vector<size_t> SLBFGS<V,M,D>:: sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng){  
    
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


template <typename V, typename M, typename D>
  V SLBFGS<V,M,D>::solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient){
    // Implementation of the standard (non-stochastic) L-BFGS solve method
    // This can be left unimplemented if only stochastic_solve is needed
    return x;
  };


  

/**
 * @brief Compute the product of the Hessian of f at x with vector v using automatic differentiation.
 *
 * 
 * @param f Function for which to compute the Hessian-vector product.
 * @param x Point at which to evaluate the Hessian.
 * @param v Vector to multiply with the Hessian.
 */
template <typename Func,typename V, typename D>
Eigen::VectorXd hessian_vector_product(Func &f, const V &weights ,const D &x, const V &v) {
    using autodiff::VectorXvar;
    using autodiff::var;


    VectorXvar w_var(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
      w_var(i) = weights(i);
    }
    
    VectorXvar v_var(v.size());
    for (int i = 0; i < v.size(); ++i) {
      v_var(i) = v(i);
    }


    var loss= f(w_var);

    VectorXvar grad = autodiff::gradient(loss, w_var);

    // Compute directional derivative of the gradient in the direction of v

    var directional = grad.dot(v_var);

    //hessian*v
    VectorXvar dugrad = autodiff::gradient(directional, w_var);
    
    Eigen::VectorXd result(dugrad.size());
    for (int i = 0; i < dugrad.size(); ++i) {
      result(i) = autodiff::val(dugrad(i));
    }

    return result;
  
};


/**
  * @brief Compute the product of the Hessian of f at weights with vector v using finite differences.
  *
  * @param g Gradient function for which to compute the Hessian-vector product.
  * @param weights Point at which to evaluate the Hessian.
  * @param x Data point (if needed by f).
  * @param v direction vector to multiply with the Hessian. 
  * @param epsilon Finite difference step size.
  * @return Hessian-vector product H*v.
  */  

  //g is expected to be a function which takes weights and data as input and returns the gradient in the third argument
template<typename Func,typename V,typename D>
V finite_difference_hvp(Func &g, const V &weights, const V &v, const D &x, double epsilon = 1e-8) {
    V w_plus = weights + epsilon * v;
    V w_minus = weights - epsilon * v;

    V grad_plus = V::Zero(weights.size());
    V grad_minus = V::Zero(weights.size());
    
    g(w_plus, x, grad_plus);
    g(w_minus, x, grad_minus);
    return (grad_plus - grad_minus) / (2.0 * epsilon);
}





/**
*@brief two-loop recursion to compute H*v where H is the inverse Hessian approximation in L-BFGS
* it should be called at each iteration to compute the search direction instead of forming H explicitly

*@param s_list List of stored displacement vectors s_k.
*@param y_list List of stored Hessian action vectors y_k.
*@param rho_list List of scalars ρ_k = 1 / (y_kᵀ s_k).
*@param v Vector to multiply with the inverse Hessian approximation.
*/
template <typename V>
V lbfgs_two_loop(const std::vector<V>& s_list, const std::vector<V>& y_list, const std::vector<double>& rho_list, const V& v) { //two loop recursion to compute H*v
    int M = s_list.size();
    std::vector<double> alpha(M);
    V q = v;

    // backward loop
    for (int i = M - 1; i >= 0; --i) {
        alpha[i] = rho_list[i] * s_list[i].dot(q);
        q = q - alpha[i] * y_list[i];
    }

    // Scaling
    double gamma = 1.0;
    if (M > 0) {
        gamma = s_list.back().dot(y_list.back()) / y_list.back().dot(y_list.back());
    }
    V r = gamma * q;

    // fowrard loop
    for (int i = 0; i < M; ++i) {
        double beta = rho_list[i] * y_list[i].dot(r);
        r = r + s_list[i] * (alpha[i] - beta);
    }
    return r;
}



/**
 * @brief compute the inverse Hessian using stored curvature pairs
 * @param r Current iteration index.
 * @param s_list List of stored displacement vectors s_k.
 * @param y_list List of stored Hessian action vectors y_k.
 * @param rho_list List of scalars ρ_k = 1 / (y_kᵀ s_k).
 * @param q Number of curvature pairs to use for the Hessian update.
 */
template <typename V>
Eigen::MatrixXd compute_inverse_Hessian(int r, const std::vector<V> &s_list, const std::vector<V> &y_list, const std::vector<double> &rho_list, int q) { //this should preserve the positive definitess of H because p_j>0
    int m = s_list.size();

    int memory=std::min(m,q);

    Eigen::MatrixXd I= Eigen::MatrixXd::Identity(s_list[0].size(), s_list[0].size());
    if (memory == 0) {
      std::cerr << "Warning: No curvature pairs stored, something went wrong in the s list assignment. Returning identity matrix as inverse Hessian approximation." << std::endl;
        return I; // Return identity if no curvature pairs are stored
    }

    double gamma= (s_list[r-1].dot(y_list[r-1])) / (y_list[r-1].dot(y_list[r-1]));
    if (gamma <= 1e-10) {
      std::cerr << "Warning: gamma is non-positive or too small, adjusting to 1.0 for stability." << std::endl;
      gamma = 1.0; // Fallback to 1.0 if gamma is too small or non-positive for stability's sake
    }
    Eigen::MatrixXd H= gamma * I;

    for (int i=r-memory; i<r; ++i){
        Eigen::MatrixXd term1= I - rho_list[i] * s_list[i]*(y_list[i].transpose());
        H = term1 * H;
        H = H * term1;
        H += rho_list[i] * s_list[i]*(s_list[i].transpose());  
      }
    return H;
}

  /**
   * @brief Perform the S-LBFGS optimization on the objective function f.
   *
   *
   * @param x Initial guess for the minimizer (passed by value).
   * @param f Objective function to minimize, mapping V → double.
   * @param Gradient Function returning the gradient ∇f(w,x), mapping (V,w) → V.
   * @param m Memory parameter: number of curvature pairs to store.
   * @param M_param THe number of pair vectors (sj,yj) to use for the Hessian update.
   * @param L The frequency of Hessian updates: every L iterations a new estimate of the Hessian is computed L=10 in the reference paper.
   * @param b Mini-batch size for stochastic gradient estimation.
   * @param b_H Mini-batch size for stochastic Hessian-vector product estimation. 10/20* b is suggested in the reference paper.
   * @param step_size Fixed step size to use in the updates.
   * @param N Total number of component functions in the finite sum loss function. i.e. = number of data points.
   * 
   * 
   * 

   * 
   * @return The final estimate of the minimizer.
   */



template <typename V, typename M, typename D>
V SLBFGS<V,M,D>::stochastic_solve(std::vector<V> x, V weights,const S_VecFun &f,const S_GradFun &S_GradFun, int m, int M_param, int L, int b, int b_H, double step_size, int N, bool verbose, int print_every) {
    int r=0;
    
    M H = M::Identity(x.size(), x.size()); //initialize H as identity

    std::vector<V> u_list;       //stored iterates for Hessian update
    std::vector<V> s_list;        // displacement u_r − u_r-1
    std::vector<V> y_list;        // Hessian action on displacement
    std::vector<double> rho_list; // Scalars ρ_k = 1 / (y_kᵀ s_k)

    //I have verified that set upt this way it generates different sequences of random numbers at each run in minibatch sampling
    int seed=56;
    std::mt19937 rng(seed); 

    int dim_weights = weights.size();

    V wt = weights;

    std::vector<V> w_history;

    while (_iters < _max_iters) { //todo add convergence check 
    
    w_history.clear();
      // Sample minibatch for gradient estimation
    V full_gradient = V::Zero(dim_weights);
    V temp_gradient= V::Zero(dim_weights);

    for (size_t i=0; i < N; ++i) {
      temp_gradient= V::Zero(dim_weights);
      S_GradFun(weights, x[i], temp_gradient);
      full_gradient += temp_gradient;
    }

    full_gradient /= N; // 1/N sum g_i(weights)

  
    wt = weights;
    w_history.push_back(wt);
    V variance_reduced_gradient = V::Zero(dim_weights);

    const bool verbose_this_epoch = verbose && (_iters == 0);
    

    for (int t=0; t < m ; ++t){

      
      auto minibatch_indices = sample_minibatch_indices(N, b, rng);
      
      V grad_estimate_wt = V::Zero(dim_weights);
      V grad_estimate_wk = V::Zero(dim_weights);
      V temp_grad_estimate_wt= V::Zero(dim_weights);
      V temp_grad_estimate_wk= V::Zero(dim_weights);
      
      for (size_t i=0; i < minibatch_indices.size(); ++i) {
        size_t idx = minibatch_indices[i];
        temp_grad_estimate_wt= V::Zero(dim_weights);
        temp_grad_estimate_wk= V::Zero(dim_weights);
        S_GradFun(wt, x[idx], temp_grad_estimate_wt);
        grad_estimate_wt += temp_grad_estimate_wt;
        S_GradFun(weights, x[idx], temp_grad_estimate_wk);
        grad_estimate_wk += temp_grad_estimate_wk;
      }
    
      grad_estimate_wt /= b;
      grad_estimate_wk /= b;

      variance_reduced_gradient = (grad_estimate_wt - grad_estimate_wk) + full_gradient;

      if (verbose_this_epoch) {
        const bool should_print = (t < 5) || (print_every > 0 && (t % print_every == 0)) || (t == m - 1);
        if (should_print) {
          double mean_loss_t = 0.0;
          for (int i = 0; i < N; ++i) {
            mean_loss_t += f(wt, x[static_cast<size_t>(i)]);
          }
          mean_loss_t /= static_cast<double>(N);
          std::cout << "  [epoch 1] t=" << t
                    << "  ||v_t||=" << variance_reduced_gradient.norm()
                    << "  mean_loss(w_t)=" << mean_loss_t << std::endl;
        }
      }

//      std::cout << "variance reduced gradient norm at inner iter " << t << ": " << variance_reduced_gradient.norm() << std::endl;

      wt = wt - step_size* lbfgs_two_loop(s_list, y_list, rho_list, variance_reduced_gradient);


      w_history.push_back(wt);

      if (t%L==0 && t>0){

        
        r++;
        
        V u = V::Zero(dim_weights);
        const int num_wt = static_cast<int>(w_history.size());
        const int start = std::max(0, num_wt - L);
        const int count = num_wt - start;
        for (int j = start; j < num_wt; ++j) {
          u += w_history[static_cast<size_t>(j)];
        }
        if (count > 0) {
          u /= static_cast<double>(count);
        }

        // costruisci s,y solo se abbiamo un u precedente
        if (!u_list.empty()) {
          const V &u_prev = u_list.back();
          V s = u - u_prev;
          s_list.push_back(s);

          auto minibatch_indices_H = sample_minibatch_indices(N, b_H, rng);
          V y = V::Zero(dim_weights);
          for (size_t i = 0; i < minibatch_indices_H.size(); ++i) {
            const size_t idx = minibatch_indices_H[i];
            y += finite_difference_hvp(S_GradFun, u, s, x[idx]);
          }
          if (!minibatch_indices_H.empty()) {
            y /= static_cast<double>(minibatch_indices_H.size());
          }

          y_list.push_back(y);
          const double ys = y.dot(s);
         if (std::abs(ys) > 1e-14) {
            rho_list.push_back(1.0 / ys);
          } else {
            s_list.pop_back();
            y_list.pop_back();}
        }

        if (M_param > 0 && s_list.size() > static_cast<size_t>(M_param)) {
          s_list.erase(s_list.begin());
          y_list.erase(y_list.begin());
          rho_list.erase(rho_list.begin());
        }

        u_list.push_back(u);
      
      //H= compute_inverse_Hessian(r, s_list, y_list, rho_list, M_param); //this step is done implicitly in the two loop recursion
    }  
  }

  if (s_list.size()!=y_list.size() || s_list.size()!=rho_list.size()){
    std::cerr<<"s_list, y_list and rho_list sizes do not match"<<std::endl;
  }


  if (w_history.size() >= 2) {
    const int max_i = std::min(m - 1, static_cast<int>(w_history.size()) - 2);
    if (max_i >= 0) {
      // Algorithm 1 (step 18): choose wk+1 = x_i, i in {0,...,m-1}.
      // In pratica, se m è piccolo, scegliere spesso i=0 resetta il progresso.
      // Qui evitiamo il reset sistematico preferendo i>=1 quando possibile.
      const int min_i = (max_i >= 1) ? 1 : 0;
      std::uniform_int_distribution<int> pick_i(min_i, max_i);
      weights = w_history[static_cast<size_t>(pick_i(rng))];
      wt = weights;
    }
  }

  // Nota: la scelta random dello step 18 sovrascrive wt.
  // Per monitorare la convergenza dell'inner loop, la loss va valutata su x_m
  // (cioè il wt prima dello step 18). Usiamo l'ultimo elemento di w_history.
  const V &x_m = w_history.back();
  double mean_loss = 0.0;
  for (int i = 0; i < N; ++i) {
    mean_loss += f(x_m, x[static_cast<size_t>(i)]);
  }
  mean_loss /= static_cast<double>(N);
  std::cout << "Iteration " << (_iters + 1) << ": Mean Loss = " << mean_loss << std::endl;




  _iters++;
}    
  return wt;
};





