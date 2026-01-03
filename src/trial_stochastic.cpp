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

auto loss = [](const Vec& w, const D& x) { return loss_fun(w, x); };
auto grad = [](const Vec& w, const D& x, Vec& g) { grad_fun(w, x, g); };

int main() {

	int n = 5; // dimensione dei vettori
	int N = 20; // numero di dati
	int m = 5, M_param = 3, L = 2, b = 4, b_H = 2;
	double step_size = 0.1;

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
	
	SLBFGS<Vec, Mat, Vec> solver;
	solver.setMaxIterations(max_iters);
	solver.setTolerance(1e-8);


	auto loss = [](const Vec& w, const Vec& x) { return loss_fun(w, x); };
	auto grad = [](const Vec& w, const Vec& x, Vec& g) { grad_fun(w, x, g); };

	// stampa loss ad ogni iterazione per vedere se converge
	Vec current_weights = weights;
	for (int iter = 0; iter < max_iters; ++iter) {
		current_weights = solver.stochastic_solve(x_data, current_weights, loss, grad, m, M_param, L, b, b_H, step_size, N);
	
		double mean_loss = 0.0;
		for (int i = 0; i < N; ++i) {
			mean_loss += loss(current_weights, x_data[i]);
		}
		mean_loss /= N;
		//std::cout << "Iterazione " << iter+1 << ": Loss media = " << mean_loss << std::endl;
	}
	std::cout << "Risultato ottimizzazione (w):\n" << current_weights.transpose() << std::endl;
	return 0;
}
