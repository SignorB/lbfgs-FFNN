#pragma once

#include "common.hpp"
#include "minimizer_base.hpp"


#include <Eigen/Eigen>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>


#include <random>
#include <numeric>
#include <fstream>
#include <cmath>



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
  using Base::stochastic_m;
  using Base::M_param;
  using Base::L;
  using Base::b;
  using Base::b_H;
  using Base::step_size;
  using Base::alpha_wolfe;
  using Base::m;



using S_VecFun=std::function<double(const V&, const V&, const V&)>; //loss function must take weights, input, and target as input and returns a double
using S_GradFun=std::function<void(const V&, const V&, const V&, V&)>; //gradient function must take weights, input, target as input and returns a vector


public:
  V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
    return stochastic_solve(_inputs, _targets, x, _sf, _sg, stochastic_m, M_param, L, b, b_H, step_size, static_cast<int>(_inputs.size()), false, 50);
  };
  V stochastic_solve(std::vector<V> inputs, std::vector<V> targets, V weights,const S_VecFun &f,const S_GradFun &S_GradFun, int m, int M_param, int L, int b, int b_H, double step_size, int N, bool verbose=false, int print_every=50);

  
  void setLogFile(const std::string &path) { _logfile = path; }
  static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);

  void setData(const std::vector<V> &inputs, const std::vector<V> &targets, const S_VecFun &f, const S_GradFun &g) {
    _inputs = inputs;
    _targets = targets;
    _sf = f;
    _sg = g;
  }

private:
  std::vector<V> _inputs;
  std::vector<V> _targets;
  S_VecFun _sf;
  S_GradFun _sg;
  std::string _logfile;

};

  //returns uniform sample of a minibatch
  /**
  * @brief Sample a minibatch of indices from 0 to N-1.
  *
  * @param N Total number of data points.
  * @param batch_size Size of the minibatch to sample.
  * @param rng Random number generator.
  */

template <typename V, typename M>
std::vector<size_t> SLBFGS<V,M>:: sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng){  
    
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

  

/**
 * @brief Compute the product of the Hessian of f at x with vector v using automatic differentiation.
 *
 * 
 * @param f Function for which to compute the Hessian-vector product.
 * @param x Point at which to evaluate the Hessian.
 * @param v Vector to multiply with the Hessian.
 */
template <typename Func,typename V>
Eigen::VectorXd hessian_vector_product(Func &f, const V &weights ,const V &input, const V &target, const V &v) { //todo, doesnt work. Autodiff problem
    using autodiff::VectorXvar;
    using autodiff::var;


    VectorXvar w_var(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
      w_var(i) = weights(i);
    }
    //debug
    std::cout<< "hvp weights conversion" << std::endl;


    VectorXvar v_var(v.size());
    for (int i = 0; i < v.size(); ++i) {
      v_var(i) = v(i);
    }

    //debug
    std::cout<< "hvp v conversion" << std::endl;


    var loss= f(w_var, input, target);

    VectorXvar grad = autodiff::gradient(loss, w_var);

    //debug
    std::cout<< "hvp gradient computed" << std::endl;

    // Compute directional derivative of the gradient in the direction of v

    var directional = grad.dot(v_var);

    //debug
    std::cout<< "hvp directional derivative computed" << std::endl;


    //hessian*v
    VectorXvar dugrad = autodiff::gradient(directional, w_var);
    
    //debug    
    std::cout<< "hvp hvp computed" << std::endl;


    Eigen::VectorXd result(dugrad.size());
    for (int i = 0; i < dugrad.size(); ++i) {
      result(i) = autodiff::val(dugrad(i));
    }

    //debyg

    return result;
  
};


/**
  * @brief Compute the product of the Hessian of f at weights with vector v using finite differences.
  *
  * @param g Gradient function for which to compute the Hessian-vector product.
  * @param weights Point at which to evaluate the Hessian.
  * @param input Input data point.
  * @param target Target data point.
  * @param v direction vector to multiply with the Hessian. 
  * @param epsilon Finite difference step size.
  * @return Hessian-vector product H*v.
  */  

  //g is expected to be a function which takes weights, input, target as input and returns the gradient in the fourth argument
template<typename Func,typename V>
V finite_difference_hvp(Func &g, const V &weights, const V &input, const V &target, const V &v, double epsilon = 1e-6) {
    V w_plus = weights + epsilon * v;
    V w_minus = weights - epsilon * v;

    V grad_plus = V::Zero(weights.size());
    V grad_minus = V::Zero(weights.size());
    
    g(w_plus, input, target, grad_plus);
    g(w_minus, input, target, grad_minus);
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
      double denom = y_list.back().dot(y_list.back());
      if (std::abs(denom) < 1e-12) {
        gamma = 1.0;
      } else {
        gamma = s_list.back().dot(y_list.back()) / denom;
      }
      gamma = std::min(std::max(gamma, 1e-6), 1e6);
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
   * @param inputs Vector of input data points.
   * @param targets Vector of target data points.
   * @param weights Initial weights.
   * @param f Objective function to minimize, mapping (V, V, V) → double.
   * @param S_GradFun Function returning the gradient ∇f(w, input, target), mapping (V, V, V) → V.
   * @param m Memory parameter: number of curvature pairs to store.
   * @param M_param The number of pair vectors (sj,yj) to use for the Hessian update.
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
template <typename V, typename M>
V SLBFGS<V,M>::stochastic_solve(std::vector<V> inputs, std::vector<V> targets, V weights,const S_VecFun &f,const S_GradFun &S_GradFun, int m, int M_param, int L, int b, int b_H, double step_size, int N, bool verbose, int print_every) {
    int r=0;
    double passes = 0.0; // number of passes through data 
    std::ofstream logfile_stream;
    if (!_logfile.empty()) {
      // Open in append mode. If file is empty/new, write header.
      bool need_header = true;
      {
        std::ifstream fin(_logfile);
        if (fin.good()) {
          int c = fin.get();
          if (c != EOF) need_header = false;
        }
      }
      logfile_stream.open(_logfile, std::ofstream::out | std::ofstream::app);
      if (logfile_stream.is_open() && need_header) {
        logfile_stream << "passes,loss,log10_loss,iteration" << std::endl;
        logfile_stream.flush();
      }
    }
    

    std::vector<V> u_list;       //stored iterates for Hessian update
    std::vector<V> s_list;        // displacement u_r − u_r-1
    std::vector<V> y_list;        // Hessian action on displacement
    std::vector<double> rho_list; // Scalars ρ_k = 1 / (y_kᵀ s_k)

    int seed=56;
    std::mt19937 rng(seed); 

    int dim_weights = weights.size();

    V wt = weights;

    // Only keep the last L+1 iterates for averaging, not all history
    std::vector<V> w_history;
    w_history.reserve(static_cast<size_t>(L + 1));

    while (_iters < _max_iters) { //convergence check based on gradient norm after computing the full gradient
     

      w_history.clear();
      // Sample minibatch for gradient estimation
    V full_gradient = V::Zero(dim_weights);
    V temp_gradient= V::Zero(dim_weights);

    for (int i=0; i < N; ++i) {
      temp_gradient.setZero();
      S_GradFun(weights, inputs[i], targets[i], temp_gradient);
      full_gradient += temp_gradient;
    }

    full_gradient /= N; // 1/N sum g_i(weights)
    // full gradient computation= one pass through data
    passes += 1.0;

    if (full_gradient.norm()<_tol){
      std::cout<<"Converged: gradient norm "<<full_gradient.norm()<<" below tolerance "<<_tol<<std::endl;
      break;
    }


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
        temp_grad_estimate_wt.setZero();
        temp_grad_estimate_wk.setZero();
        S_GradFun(wt, inputs[idx], targets[idx], temp_grad_estimate_wt);
        grad_estimate_wt += temp_grad_estimate_wt;
        S_GradFun(weights, inputs[idx], targets[idx], temp_grad_estimate_wk);
        grad_estimate_wk += temp_grad_estimate_wk;
      }
    
      grad_estimate_wt /= b;
      grad_estimate_wk /= b;

      variance_reduced_gradient = (grad_estimate_wt - grad_estimate_wk) + full_gradient;

      // Two gradient estimates per minibatch (wt and weights) each using b samples
      passes += (2.0 * static_cast<double>(b)) / static_cast<double>(N);

      /*
      if (verbose_this_epoch) {
        const bool should_print = (t < 5) || (print_every > 0 && (t % print_every == 0)) || (t == m - 1);
        if (should_print) {
          double mean_loss_t = 0.0;
          for (int i = 0; i < N; ++i) {
            mean_loss_t += f(wt, inputs[static_cast<size_t>(i)], targets[static_cast<size_t>(i)]);
          }
          mean_loss_t /= static_cast<double>(N);
          std::cout << "  [epoch 1] t=" << t
          << "  ||v_t||=" << variance_reduced_gradient.norm()
          << "  mean_loss(w_t)=" << mean_loss_t << std::endl;
        }
      }
      */


      wt = wt - step_size * lbfgs_two_loop(s_list, y_list, rho_list, variance_reduced_gradient);
      // Only keep last L+1 iterates to limit memory usage
      if (w_history.size() >= static_cast<size_t>(L + 1)) {
        w_history.erase(w_history.begin());
      }
      w_history.push_back(wt);

      if (t%L==0 && t>0){

        
        r++;
        
        V u = V::Zero(dim_weights);
        const int num_wt = static_cast<int>(w_history.size());
        for (int j = 0; j < num_wt; ++j) {
          u += w_history[static_cast<size_t>(j)];
        }
        if (num_wt > 0) {
          u /= static_cast<double>(num_wt);
        }

        if (!u_list.empty()) {
          const V &u_prev = u_list.back();
          V s = u - u_prev;
          s_list.push_back(s);

          auto minibatch_indices_H = sample_minibatch_indices(N, b_H, rng);
          V y = V::Zero(dim_weights);
          for (size_t i = 0; i < minibatch_indices_H.size(); ++i) {
            const size_t idx = minibatch_indices_H[i];
            y += finite_difference_hvp(S_GradFun, u, inputs[idx], targets[idx], s);
          }
          if (!minibatch_indices_H.empty()) {
            y /= static_cast<double>(minibatch_indices_H.size());
          }

          y_list.push_back(y);
          // Each finite-difference HVP costs 2 gradient evaluations per sample
          passes += (2.0 * static_cast<double>(minibatch_indices_H.size())) / static_cast<double>(N);
          const double ys = y.dot(s);
         if (std::abs(ys) > 1e-12) {
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

        // Only keep last few u values needed
        if (u_list.size() >= static_cast<size_t>(M_param + 1)) {
          u_list.erase(u_list.begin());
        }
        u_list.push_back(u);
      
    }  
  }

  if (s_list.size()!=y_list.size() || s_list.size()!=rho_list.size()){
    std::cerr<<"s_list, y_list and rho_list sizes do not match"<<std::endl;
  }


  if (w_history.size() >= 2) {
    std::uniform_int_distribution<size_t> pick_i(0, w_history.size() - 2);
    weights = w_history[pick_i(rng)];
    wt = weights;
  }



  const V &x_m = w_history.back();
  double loss = 0.0;
  for (int i = 0; i < N; ++i) {
    loss += f(x_m, inputs[static_cast<size_t>(i)], targets[static_cast<size_t>(i)]);
  }
 // mean_loss /= static_cast<double>(N);
  std::cout << "Iteration " << (_iters + 1) << ": Loss = " << loss << std::endl;

  if (logfile_stream.is_open()) {
    double log_loss = (loss > 0.0) ? std::log10(loss) : -INFINITY;
    logfile_stream << passes << "," << loss << "," << log_loss << "," << (_iters + 1) << std::endl;
    logfile_stream.flush();
  }




  _iters++;
}    

  if (logfile_stream.is_open()) logfile_stream.close();

  return wt;
};





