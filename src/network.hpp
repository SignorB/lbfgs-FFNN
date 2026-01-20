#pragma once
#include "layer.hpp"
#include <Eigen/Core>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#include "minimizer_base.hpp"
#include "s_lbfgs.hpp"
#include "s_gd.hpp"
#include <iostream>

class Network {
private:
  std::vector<std::unique_ptr<Layer>> layers;

  std::vector<double> params;
  std::vector<double> grads;
  size_t params_size = 0;

  std::vector<Eigen::MatrixXd> activations;
  std::vector<Eigen::MatrixXd> deltas;

public:
  Network() = default;

  size_t getSize() const {
    return params_size;
  }

  template <int In, int Out, typename Activation = Linear>
  void addLayer() {
    layers.push_back(std::make_unique<DenseLayer<In, Out, Activation>>());
    params_size += layers.back()->getParamsSize();
  }

  void bindParams() {
    if (layers.empty())
      return;

    params.resize(params_size);
    grads.resize(params_size);

    std::random_device rd;
    std::mt19937 gen(rd());

    activations.clear();
    deltas.clear();

    double *p_ptr = params.data();
    double *g_ptr = grads.data();

    for (auto &layer : layers) {
      layer->bind(p_ptr, g_ptr);
      double std_dev = layer->getInitStdDev();
      std::normal_distribution<double> dist(0.0, std_dev);
      for (int i = 0; i < layer->getParamsSize(); ++i) {
        p_ptr[i] = dist(gen);
      }
      p_ptr += layer->getParamsSize();
      g_ptr += layer->getParamsSize();
    }
  }

  const Eigen::MatrixXd &forward(const Eigen::MatrixXd &input) {
    if (activations.empty() || activations[0].cols() != input.cols()) {
        activations.resize(layers.size() + 1);
    }
    
    activations[0] = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
      layers[i]->forward(activations[i], activations[i + 1]);
    }

    return activations.back();
  }

  void backward(const Eigen::MatrixXd &loss_grad) {
    if (deltas.size() != layers.size() + 1) {
        deltas.resize(layers.size() + 1);
    }

    deltas.back() = loss_grad;

    for (int i = layers.size() - 1; i >= 0; --i) {
      layers[i]->backward(
          deltas[i + 1],
          (i > 0) ? &deltas[i] : nullptr);
    }
  }

  void zeroGrads() {
    std::fill(grads.begin(), grads.end(), 0.0);
  }

  double *getParamsData() { return params.data(); }
  double *getGradsData() { return grads.data(); }

  void setParams(const Eigen::VectorXd &new_params) {
    std::copy(new_params.data(), new_params.data() + params_size, params.begin());
  }

  void getGrads(Eigen::VectorXd &out_grads) {
    std::copy(grads.begin(), grads.end(), out_grads.data());
  }

  void train(const Eigen::MatrixXd &inputs,
             const Eigen::MatrixXd &targets,
             std::shared_ptr<MinimizerBase<Eigen::VectorXd, Eigen::MatrixXd>> minimizer) {

    Eigen::VectorXd x(params_size);
    std::copy(params.begin(), params.end(), x.data());

    VecFun<Eigen::VectorXd, double> f = [&](const Eigen::VectorXd &p) -> double {
      this->setParams(p);
      const auto &output = this->forward(inputs);
      Eigen::MatrixXd diff = output - targets;
      double total_loss = 0.5 * diff.squaredNorm();
      total_loss /= inputs.cols(); // Normalize by N
      // std::cout << "MSE: " << total_loss << std::endl; // removed to reduce spam
      return total_loss;
    };

    GradFun<Eigen::VectorXd> g = [&](const Eigen::VectorXd &p) -> Eigen::VectorXd {
      this->setParams(p);
      this->zeroGrads();

      const auto &output = this->forward(inputs);
      Eigen::MatrixXd loss_grad = output - targets;
      loss_grad /= inputs.cols(); // Normalize gradients by N

      this->backward(loss_grad);

      Eigen::VectorXd grad_vec(params_size);
      this->getGrads(grad_vec);
      return grad_vec;
    };

    Eigen::VectorXd final_params = minimizer->solve(x, f, g);
    this->setParams(final_params);
  }

  void test(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, std::string label = "Test Results") {
    const auto &output = this->forward(inputs);
    long correct = 0;
    long total = inputs.cols();

    for (long i = 0; i < total; ++i) {
        Eigen::Index pred_idx, true_idx;
        output.col(i).maxCoeff(&pred_idx);
        targets.col(i).maxCoeff(&true_idx);
        if (pred_idx == true_idx) {
            correct++;
        }
    }

    double accuracy = (double)correct / total * 100.0;
    Eigen::MatrixXd diff = output - targets;
    double mse = 0.5 * diff.squaredNorm();

    std::cout << "=== " << label << " ===" << std::endl;
    std::cout << "Samples: " << total << std::endl;
    std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")" << std::endl;
    std::cout << "Total MSE: " << mse << std::endl;
    std::cout << "====================" << std::endl;
  }


  void train_stochastic(const Eigen::MatrixXd &inputs,
                        const Eigen::MatrixXd &targets,
                        std::shared_ptr<SLBFGS<Eigen::VectorXd, Eigen::MatrixXd>> minimizer,
                        int m, int M_param, int L, int b, int b_H, double step_size,
                        bool verbose, int print_every = 50) {
    
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    // Extract initial weights
    Vec weights(params_size);
    std::copy(params.begin(), params.end(), weights.data());
    double lambda = 1e-4; // L2 regularization coefficient

    long input_rows = inputs.rows();
    long output_rows = targets.rows();
    int N = inputs.cols();

    // Reusable buffers
    // Max capacity needed is usually max(b, b_H) or N if full gradient
    // We'll resize dynamically in callback to be safe, but pre-reserve helps
    Mat batch_x_buffer(input_rows, std::max(b, 128));
    Mat batch_y_buffer(output_rows, std::max(b, 128));

    // Batch Gradient Callback
    auto batch_g = [&](const Vec &w, const std::vector<size_t>& indices, Vec &grad) mutable {
        this->setParams(w);
        this->zeroGrads();
        
        long current_bs = indices.size();
        
        // Resize logical view
        if(batch_x_buffer.cols() < current_bs) {
            batch_x_buffer.resize(input_rows, current_bs);
            batch_y_buffer.resize(output_rows, current_bs);
        }
        
        // Gather
        // If full batch (indices 0..N-1), avoiding copy if possible would be nice, 
        // but 'inputs' is passed by reference, so we can't just point to it 
        // unless we overload forward/backward to take Matrix Map/Slice. 
        // For now, copy is still much faster than single-sample iteration.
        // Optimization: Check if indices are sequential 0..N-1, use full inputs directly?
        bool is_full_batch = (current_bs == N); // Approximate check
        
        const Mat* x_ptr = &batch_x_buffer;
        
        if (is_full_batch) {
             // If we are trusting valid indices 0..N-1
             this->forward(inputs);
             Mat diff = activations.back() - targets;
             this->backward(diff);
        } else {
             for(size_t i=0; i < current_bs; ++i) {
                 batch_x_buffer.col(i) = inputs.col(indices[i]);
                 batch_y_buffer.col(i) = targets.col(indices[i]);
             }
             // Forward on batch subset
             // We need to use left-block of buffer
             auto x_view = batch_x_buffer.leftCols(current_bs);
             auto y_view = batch_y_buffer.leftCols(current_bs);

             this->forward(x_view);
             Mat diff = activations.back() - y_view;
             this->backward(diff);
        }

        this->getGrads(grad);
        
        // Average the gradient
        grad /= static_cast<double>(current_bs);
        
        // L2 Regularization
        grad.array() += lambda * w.array();
    };


    // Batch Loss Callback (Optional, for logging)
    auto batch_f = [&](const Vec &w, const std::vector<size_t>& indices) -> double {
        this->setParams(w);
        long current_bs = indices.size();
        
         if(batch_x_buffer.cols() < current_bs) {
            batch_x_buffer.resize(input_rows, current_bs);
            batch_y_buffer.resize(output_rows, current_bs);
        }
        
        for(size_t i=0; i < current_bs; ++i) {
             batch_x_buffer.col(i) = inputs.col(indices[i]);
             batch_y_buffer.col(i) = targets.col(indices[i]);
        }
         auto x_view = batch_x_buffer.leftCols(current_bs);
         auto y_view = batch_y_buffer.leftCols(current_bs);
        
        const auto &output = this->forward(x_view);
        Vec diff_sq = (output - y_view).colwise().squaredNorm();
        double loss = 0.5 * diff_sq.sum();
        loss /= current_bs;
        loss += 0.5 * lambda * w.squaredNorm();
        return loss;
    };

    // Set Callbacks
    minimizer->setData(batch_f, batch_g);

    Vec final_weights = minimizer->stochastic_solve(
        weights,
        batch_f,
        batch_g,
        m, 
        M_param, 
        L, 
        b, 
        b_H, 
        step_size, 
        N, 
        verbose, 
        print_every
    );
    this->setParams(final_weights);
  }

  // Optimized Method for Batch SGD
  void train_sgd(const Eigen::MatrixXd &inputs,
                 const Eigen::MatrixXd &targets,
                 std::shared_ptr<StochasticGradientDescent<Eigen::VectorXd, Eigen::MatrixXd>> minimizer,
                 int m, int batch_size, double step_size,
                 bool verbose, int print_every = 50) {
      
      using Vec = Eigen::VectorXd;
      using Mat = Eigen::MatrixXd;

      // Extract initial weights
      Vec weights(params_size);
      std::copy(params.begin(), params.end(), weights.data());

      // Pre-allocate batch buffers (reused)
      // Capacity = batch_size
      long input_rows = inputs.rows();
      long output_rows = targets.rows();
      
      // Thread-local or lambda-captured buffers to avoid reallocation
      Mat batch_x_buffer(input_rows, batch_size);
      Mat batch_y_buffer(output_rows, batch_size);

      // Gradient Callback (Batch-Optimized)
      auto batch_g = [&](const Vec &w, const std::vector<size_t>& indices, Vec &grad) mutable {
          this->setParams(w);
          this->zeroGrads();
          
          long current_bs = indices.size();

          // Resize logical view if last batch is smaller (avoids realloc if capacity is sufficient)
          if(batch_x_buffer.cols() != current_bs) {
              batch_x_buffer.resize(input_rows, current_bs);
              batch_y_buffer.resize(output_rows, current_bs);
          }

          // Gather: Copy columns from main data to contiguous batch buffer
          // This is the cost paid to enable fast Matrix-Matrix multiply next
          for(size_t i=0; i < current_bs; ++i) {
              batch_x_buffer.col(i) = inputs.col(indices[i]);
              batch_y_buffer.col(i) = targets.col(indices[i]);
          }
          
          // Forward Pass (Matrix Operation) - This is where speedup comes from
          const auto &output = this->forward(batch_x_buffer);
          
          // Backward Pass
          Mat diff = output - batch_y_buffer;
          this->backward(diff); 
          
          // Accumulate Gradients
          this->getGrads(grad);

          // Average gradient over batch
          grad /= static_cast<double>(current_bs);
      };

      // Loss Callback (Single sample for logging compatibility, or small batch if desired)
      // Note: Evaluating loss on full dataset every epoch is SLOW. 
      // Current s_gd implementation calls this N times. 
      // We keep it simple here.
      auto f_single = [&](const Vec &w, const Vec &x, const Vec &y) -> double {
           this->setParams(w);
           // Adapter for single vector
           Eigen::MatrixXd input_mat(x.size(), 1);
           input_mat.col(0) = x;
           const auto &output = this->forward(input_mat);
           return 0.5 * (output.col(0) - y).squaredNorm();
      };

      minimizer->setData(inputs, targets, f_single, batch_g);

      Vec final_weights = minimizer->stochastic_solve(
          weights,
          m, 
          batch_size, 
          step_size, 
          verbose, 
          print_every
      );
      
      this->setParams(final_weights);
  }
};