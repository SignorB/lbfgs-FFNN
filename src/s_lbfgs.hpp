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
 *
 * Use "Batch Processing": Delegates data handling to a BatchGradFun callback.
 * This allows the caller (Network) to perform efficient Matrix-Matrix operations
 * for gradients instead of iterating sample-by-sample.
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

    // Callbacks now take a list of indices instead of raw data
    // Caller is responsible for mapping indices -> data -> gradient
    using BatchGradFun = std::function<void(const V&, const std::vector<size_t>&, V&)>; 
    using BatchLossFun = std::function<double(const V&, const std::vector<size_t>&)>;

public:
    V solve(V x, VecFun<V, double> &f, GradFun<V> &Gradient) override {
        // Fallback or Error: This minimizer requires Batch Callbacks.
        // For compatibility with MinimizerBase, we could wrap full-batch calls, 
        // but typically we call stochastic_solve directly.
        throw std::runtime_error("SLBFGS::solve(x, f, g) not supported. Use stochastic_solve with batch callbacks.");
        return x;
    };

    /**
     * @brief Stochastic Solve with Batch Callbacks
     * 
     * @param weights Initial weights
     * @param f Loss function callback (takes batch indices)
     * @param batch_g Gradient function callback (takes batch indices)
     * @param N Total dataset size (used to generate indices)
     */
    V stochastic_solve(V weights, 
                       const BatchLossFun &f, 
                       const BatchGradFun &batch_g, 
                       int m, int M_param, int L, int b, int b_H, double step_size, int N, 
                       bool verbose=false, int print_every=50);

    void setLogFile(const std::string &path) { _logfile = path; }
    
    // Helper to generate random indices
    static std::vector<size_t> sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng);

    // Legacy setData removed/deprecated as we pass functors directly to solve or rely on caller context
    void setData(const BatchLossFun &f, const BatchGradFun &g) {
        _sf = f;
        _sg = g;
    }

private:
    BatchLossFun _sf;
    BatchGradFun _sg;
    std::string _logfile;
};

// -------------------------------------------------------------------------
// Helper: Finite Difference HVP on a Batch
// -------------------------------------------------------------------------
template<typename V, typename BatchFn>
V finite_difference_hvp_batch(BatchFn &g, const V &weights, const std::vector<size_t>& indices, const V &v, double epsilon = 1e-4) {
    V w_plus = weights + epsilon * v;
    V w_minus = weights - epsilon * v;

    V grad_plus = V::Zero(weights.size());
    V grad_minus = V::Zero(weights.size());
    
    // Compute gradients on the SAME batch for perturbed weights
    g(w_plus, indices, grad_plus);
    g(w_minus, indices, grad_minus);

    return (grad_plus - grad_minus) / (2.0 * epsilon);
}

// -------------------------------------------------------------------------
// Helper: L-BFGS Two Loop Recursion
// -------------------------------------------------------------------------
template <typename V>
V lbfgs_two_loop(const std::vector<V>& s_list, const std::vector<V>& y_list, const std::vector<double>& rho_list, const V& v) {
    int M = s_list.size();
    std::vector<double> alpha(M);
    V q = v;

    // Backward
    for (int i = M - 1; i >= 0; --i) {
        alpha[i] = rho_list[i] * s_list[i].dot(q);
        q = q - alpha[i] * y_list[i];
    }

    // Scaling
    double gamma = 1.0;
    if (M > 0) {
      double denom = y_list.back().dot(y_list.back());
      if (std::abs(denom) < 1e-12) gamma = 1.0;
      else gamma = s_list.back().dot(y_list.back()) / denom;
      gamma = std::min(std::max(gamma, 1e-6), 1e6);
    }
    V r = gamma * q;

    // Forward
    for (int i = 0; i < M; ++i) {
        double beta = rho_list[i] * y_list[i].dot(r);
        r = r + s_list[i] * (alpha[i] - beta);
    }
    return r;
}

// -------------------------------------------------------------------------
// Implementation: sample_minibatch_indices
// -------------------------------------------------------------------------
template <typename V, typename M>
std::vector<size_t> SLBFGS<V,M>::sample_minibatch_indices(const size_t N, size_t batch_size, std::mt19937 &rng){  
    if (N == 0 || batch_size == 0) return {};
    if (batch_size >= N) {
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }
    
    // Partial Fisher-Yates
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t i = 0; i < batch_size; ++i) {
      std::uniform_int_distribution<size_t> dist(i, N -1);
      size_t j = dist(rng);
      std::swap(idx[i], idx[j]);
    }
    idx.resize(batch_size);
    return idx;
}

// -------------------------------------------------------------------------
// Implementation: stochastic_solve
// -------------------------------------------------------------------------
template <typename V, typename M>
V SLBFGS<V,M>::stochastic_solve(V weights, 
                                const BatchLossFun &f, 
                                const BatchGradFun &batch_g, 
                                int m, int M_param, int L, int b, int b_H, double step_size, int N, 
                                bool verbose, int print_every) {
    
    _iters = 0;
    double passes = 0.0;
    std::ofstream logfile_stream;
    if (!_logfile.empty()) {
       logfile_stream.open(_logfile, std::ofstream::out | std::ofstream::app);
       if(logfile_stream.is_open()) logfile_stream << "passes,loss,iteration" << std::endl;
    }
    
    std::vector<V> u_list;       
    std::vector<V> s_list;        
    std::vector<V> y_list;        
    std::vector<double> rho_list; 

    int seed=56;
    std::mt19937 rng(seed); 

    int dim_weights = weights.size();
    V wt = weights;

    std::vector<V> w_history;
    w_history.reserve(static_cast<size_t>(L + 1));

    // Full Batch Indices
    std::vector<size_t> full_indices(N);
    std::iota(full_indices.begin(), full_indices.end(), 0);

    while (_iters < _max_iters) {
        
        w_history.clear();
        
        // -------------------------------------------------------
        // 1. Compute Full Gradient (Variance Reduction Anchor)
        // -------------------------------------------------------
        V full_gradient = V::Zero(dim_weights);
        
        // Call batch gradient with ALL indices -> Calls Optimized Matrix Forward/Backward once
        batch_g(weights, full_indices, full_gradient);
        
        // Note: The callback usually averages. If it sums, we divide. 
        // Based on established convention in train_sgd, it averages.
        // We assume batch_g returns the AVERAGE gradient over indices.
        
        passes += 1.0;

        if (full_gradient.norm() < _tol){
            std::cout << "Converged: gradient norm " << full_gradient.norm() << std::endl;
            break;
        }

        wt = weights;
        w_history.push_back(wt);
        V variance_reduced_gradient = V::Zero(dim_weights);

        // -------------------------------------------------------
        // 2. Inner Loop (Stochastic Updates)
        // -------------------------------------------------------
        for (int t=0; t < m ; ++t){
            
            auto minibatch_indices = sample_minibatch_indices(N, b, rng);
            
            V grad_estimate_wt = V::Zero(dim_weights);
            V grad_estimate_wk = V::Zero(dim_weights);
            
            // Batch Gradients on Subset
            batch_g(wt, minibatch_indices, grad_estimate_wt);
            batch_g(weights, minibatch_indices, grad_estimate_wk);
            
            // SVRG Update Rule: g_eff = g(wt) - g(w_anchor) + full_grad
            variance_reduced_gradient = (grad_estimate_wt - grad_estimate_wk) + full_gradient;

            // Two gradient evals per minibatch
            passes += (2.0 * static_cast<double>(b)) / static_cast<double>(N);

            // Update Direction using L-BFGS Two Loop
            V direction = lbfgs_two_loop(s_list, y_list, rho_list, variance_reduced_gradient);
            wt = wt - step_size * direction;

            if (w_history.size() >= static_cast<size_t>(L + 1)) {
                w_history.erase(w_history.begin());
            }
            w_history.push_back(wt);

            // -------------------------------------------------------
            // 3. Hessian Update (Curvature Pairs)
            // -------------------------------------------------------
            // According to Moritz et al., update every L steps
            if (t > 0 && t % L == 0){
                
                // Compute Average iterate u
                V u = V::Zero(dim_weights);
                const int num_wt = static_cast<int>(w_history.size());
                for (const auto& w : w_history) u += w;
                if (num_wt > 0) u /= static_cast<double>(num_wt);

                if (!u_list.empty()) {
                    const V &u_prev = u_list.back();
                    V s = u - u_prev; // s = difference in average iterates

                    auto batch_indices_H = sample_minibatch_indices(N, b_H, rng);
                    
                    // Estimate y = H * s using finite differences on batch
                    V y = finite_difference_hvp_batch(batch_g, u, batch_indices_H, s);
                    
                    // HVP costs 2 batch grads
                    passes += (2.0 * static_cast<double>(b_H)) / static_cast<double>(N);

                    double ys = y.dot(s);
                    if (std::abs(ys) > 1e-10) { // Stability check
                         s_list.push_back(s);
                         y_list.push_back(y);
                         rho_list.push_back(1.0 / ys);
                    }
                }

                // Memory Limit
                if (M_param > 0 && s_list.size() > static_cast<size_t>(M_param)) {
                   s_list.erase(s_list.begin());
                   y_list.erase(y_list.begin());
                   rho_list.erase(rho_list.begin());
                }

                if (u_list.size() >= static_cast<size_t>(M_param + 1)) {
                   u_list.erase(u_list.begin());
                }
                u_list.push_back(u);
            }
        } // end inner loop

        // Reset anchor
        if (w_history.size() >= 2) {
             std::uniform_int_distribution<size_t> pick_i(0, w_history.size() - 2);
             weights = w_history[pick_i(rng)]; // Choose random iterate as new anchor? Or last? Paper says random or average.
        } else {
             weights = wt;
        }

        // Logging (approximate loss via simple full gradient norm or callback if desired)
        // Calculating full loss is expensive. We skip it or use passed callback on full batch.
         if (verbose) {
           double full_loss = f(weights, full_indices);
           std::cout << "SLBFGS Iter " << (_iters + 1) << " Loss: " << full_loss << std::endl;
         }

        _iters++;
    }

    if (logfile_stream.is_open()) logfile_stream.close();
    return weights;
};





