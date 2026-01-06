#include <Eigen/Eigen>
#include <functional>
#include <random>
#include <iostream>

#include "s_lbfgs.hpp"


using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;


// Funzione di loss quadratica: f(w, x) = 0.5 * ||w - x||^2
double loss_fun(const Vec& w, const Vec& x) {
	return 0.5 * (w - x).squaredNorm();
}

// Gradiente della loss rispetto a w: grad = w - x
void grad_fun(const Vec& w, const Vec& x, Vec& grad) {
	grad = w - x;
}

// Funzione di loss L1: f(w, x) = ||w - x||_1
double loss_fun_l1(const Vec& w, const Vec& x) {
	return (w - x).lpNorm<1>();
}

// Gradiente della loss L1 rispetto a w: grad = sign(w - x)
void grad_fun_l1(const Vec& w, const Vec& x, Vec& grad) {
	grad = (w - x).cwiseSign();
}

// Funzione di loss quadratica con regolarizzazione L2: f(w, x) = 0.5 * ||w - x||^2 + (lambda / N) * ||w||^2
double loss_fun_reg(const Vec& w, const Vec& x, double lambda, int N) {
	return 0.5 * (w - x).squaredNorm() + (lambda / N) * w.squaredNorm();
}

// Gradiente della loss con regolarizzazione L2 rispetto a w: grad = w - x + (2 * lambda / N) * w
void grad_fun_reg(const Vec& w, const Vec& x, Vec& grad, double lambda, int N) {
	grad = w - x + (2.0 * lambda / N) * w;
}

int main(void) {

	int n = 1097; // dimensione dei vettori
	int N = 2025; // numero di dati
	int M_param = 10, L = 10, b = 20, b_H = 10;
	int m=N/b; 
	double step_size = 0.01;

	// generatore dati casuali
	std::vector<Vec> x_data(N);
	std::mt19937 rng(12);
	std::uniform_real_distribution<double> dist(0.0, 4.0);
	for (int i = 0; i < N; ++i) {
		x_data[i] = Vec::NullaryExpr(n, [&]() { return dist(rng); });
	}

	// Inizializza i pesi casualmente
	Vec weights = Vec::NullaryExpr(n, [&]() { return dist(rng); });

	int max_iters = 20;


	auto loss = [&](const Vec& w, const Vec& x) { return loss_fun_reg(w, x, 0.1, N); };
	auto grad = [&](const Vec& w, const Vec& x, Vec& g) { grad_fun_reg(w, x, g, 0.1, N); };

	// Loss iniziale (baseline)
	double mean_loss0 = 0.0;
	for (int i = 0; i < N; ++i) {
		mean_loss0 += loss(weights, x_data[i]);
	}
	mean_loss0 /= static_cast<double>(N);
	std::cout << "Initial: Mean Loss = " << mean_loss0 << std::endl;

	Vec w_star = Vec::Zero(n);
	for (int i = 0; i < N; ++i) {
		w_star += x_data[i];
	}
	w_star /= static_cast<double>(N);
	double mean_loss_star = 0.0;
	for (int i = 0; i < N; ++i) {
		mean_loss_star += loss(w_star, x_data[i]);
	}
	mean_loss_star /= static_cast<double>(N);
	std::cout << "Theoretical optimum for the quadratic loss: Mean Loss = " << mean_loss_star << std::endl;

	Vec current_weights = weights;

	{
		SLBFGS<Vec, Mat> solver;
		solver.setMaxIterations(max_iters);
		solver.setTolerance(1e-8);
		std::vector<Vec> inputs = x_data; // Since input is not used, set to x_data
		std::vector<Vec> targets = x_data; // Target is x_data
		auto f = [&](const Vec &w, const Vec &input, const Vec &target) -> double {
			return loss(w, target);
		};
		auto g = [&](const Vec &w, const Vec &input, const Vec &target, Vec &grad) {
			grad_fun_reg(w, target, grad, 0.1, N);
		};
		solver.setData(inputs, targets, f, g);
		solver.setStochasticParams(m, M_param, L, b, b_H, 0.1);
		// Dummy f and g for base class compatibility
		VecFun<Vec, double> dummy_f = [&](Vec w) { return 0.0; };
		GradFun<Vec> dummy_g = [&](Vec w) { return Vec::Zero(w.size()); };
		current_weights = solver.solve(current_weights, dummy_f, dummy_g);  // Now uses stochastic_solve internally
	}

	if (n<25)
	std::cout << "Risultato ottimizzazione (w):\n" << current_weights.transpose() << std::endl;
	return 0;
}
