#include <Eigen/Eigen>
#include <functional>
#include <random>
#include <iostream>

#include "s_lbfgs.hpp"


using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using D=Vec; // could be any data structure, used Vec for simplicity


// Funzione di loss quadratica: f(w, x) = 0.5 * ||w - x||^2
double loss_fun(const Vec& w, const D& x) {
	return 0.5 * (w - x).squaredNorm();
}

// Gradiente della loss rispetto a w: grad = w - x
void grad_fun(const Vec& w, const D& x, Vec& grad) {
	grad = w - x;
}

// Funzione di loss L1: f(w, x) = ||w - x||_1
double loss_fun_l1(const Vec& w, const D& x) {
	return (w - x).lpNorm<1>();
}

// Gradiente della loss L1 rispetto a w: grad = sign(w - x)
void grad_fun_l1(const Vec& w, const D& x, Vec& grad) {
	grad = (w - x).cwiseSign();
}

// Funzione di loss quadratica con regolarizzazione L2: f(w, x) = 0.5 * ||w - x||^2 + (lambda / N) * ||w||^2
double loss_fun_reg(const Vec& w, const D& x, double lambda, int N) {
	return 0.5 * (w - x).squaredNorm() + (lambda / N) * w.squaredNorm();
}

// Gradiente della loss con regolarizzazione L2 rispetto a w: grad = w - x + (2 * lambda / N) * w
void grad_fun_reg(const Vec& w, const D& x, Vec& grad, double lambda, int N) {
	grad = w - x + (2.0 * lambda / N) * w;
}

int main() {

	int n = 1000; // dimensione dei vettori
	int N = 2020; // numero di dati
	int M_param = 10, L = 10, b = 20, b_H = 10;
	int m=N/b; 
	double step_size = 0.01;

	// generatore dati casuali
	std::vector<Vec> x_data(N);
	std::mt19937 rng(12);
	std::normal_distribution<double> dist(0.0, 4.0);
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
		SLBFGS<Vec, Mat, Vec> solver;
		solver.setMaxIterations(max_iters);
		solver.setTolerance(1e-8);
		current_weights = solver.stochastic_solve(x_data, current_weights, loss, grad, m, M_param, L, b, b_H, 0.1, N, false, 50);
	}

	if (n<25)
	std::cout << "Risultato ottimizzazione (w):\n" << current_weights.transpose() << std::endl;
	return 0;
}
