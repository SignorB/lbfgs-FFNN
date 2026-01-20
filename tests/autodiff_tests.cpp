#include "test.hpp"
#include <unsupported/Eigen/IterativeSolvers>

#include "../src/minimizer/bfgs.hpp"
#include "../src/common.hpp"
#include "../src/minimizer/gd.hpp"
#include "../src/minimizer/lbfgs.hpp"
#include "../src/minimizer/newton.hpp"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using minimizerPtr = std::shared_ptr<MinimizerBase<Vec, Mat>>;

// Utility: compute gradient from an autodiff objective
static Vec autodiff_grad(VecFun<autodiff::VectorXvar, autodiff::var> &f_ad, const Vec &x) {
  autodiff::VectorXvar x_var = x.cast<autodiff::var>();
  autodiff::var y = f_ad(x_var);
  return autodiff::gradient(y, x_var);
}

static double autodiff_val(VecFun<autodiff::VectorXvar, autodiff::var> &f_ad, const Vec &x) {
  autodiff::VectorXvar x_var = x.cast<autodiff::var>();
  autodiff::var y = f_ad(x_var);
  return autodiff::reverse::detail::val(y);
}

static Mat autodiff_hessian(VecFun<autodiff::VectorXvar, autodiff::var> &f_ad, const Vec &x) {
  autodiff::VectorXvar x_var = x.cast<autodiff::var>();
  Eigen::MatrixXd H;
  Eigen::VectorXd g;
  autodiff::var y = f_ad(x_var);
  H = autodiff::hessian(y, x_var, g);
  return H.cast<double>();
}

void test_rastrigin(minimizerPtr &solver) {
  VecFun<autodiff::VectorXvar, autodiff::var> f_ad = [](autodiff::VectorXvar v) {
    autodiff::var val = 0.0;
    const double A = 10.0;
    const int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
      val += (v(i) * v(i)) - (A * cos(2.0 * M_PI * v(i)));
    }
    return A * n + val;
  };

  const int n = 100;
  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = (i % 2 == 0) ? 4.0 : -4.0;

  Mat m = Mat::Identity(n, n);

  solver->setMaxIterations(3000);
  solver->setTolerance(1.e-8);
  solver->setInitialHessian(m);
  HessFun<Vec, Mat> hfun = [&](const Vec &x_in) { return autodiff_hessian(f_ad, x_in); };
  solver->setHessian(hfun);

  Vec result = solver->solve(v, f_ad);
  Vec g = autodiff_grad(f_ad, result);
  double f0 = autodiff_val(f_ad, v);
  double f1 = autodiff_val(f_ad, result);
  double g0 = autodiff_grad(f_ad, v).norm();
  double g1 = g.norm();

  // Known global minimum at the origin
  Vec expected_min = Vec::Zero(n);
  double dist = (result - expected_min).norm();
  Tests::TestSuite<Vec, Mat>::printStatus(f0, f1, g0, g1, dist, expected_min, {1e-6, 1e-6, 1e-4});
}

void test_rosenbrock(minimizerPtr &solver) {
  VecFun<autodiff::VectorXvar, autodiff::var> f_ad = [](autodiff::VectorXvar v) {
    autodiff::var val = 0.0;
    const int n = static_cast<int>(v.size());
    for (int i = 0; i < n - 1; ++i) {
      autodiff::var term1 = v(i + 1) - v(i) * v(i);
      autodiff::var term2 = 1.0 - v(i);
      val += 100.0 * term1 * term1 + term2 * term2;
    }
    return val;
  };

  const int n = 4;
  Vec v(n);
  for (int i = 0; i < n; ++i)
    v(i) = (i % 2 == 0) ? -1.2 : 1.0;

  Mat m = Mat::Identity(n, n);

  solver->setMaxIterations(5000);
  solver->setTolerance(1.e-10);
  solver->setInitialHessian(m);
  HessFun<Vec, Mat> hfun = [&](const Vec &x_in) { return autodiff_hessian(f_ad, x_in); };
  solver->setHessian(hfun);

  Vec result = solver->solve(v, f_ad);
  Vec g = autodiff_grad(f_ad, result);
  double f0 = autodiff_val(f_ad, v);
  double f1 = autodiff_val(f_ad, result);
  double g0 = autodiff_grad(f_ad, v).norm();
  double g1 = g.norm();

  Vec expected_min = Vec::Ones(n);
  double dist = (result - expected_min).norm();
  Tests::TestSuite<Vec, Mat>::printStatus(f0, f1, g0, g1, dist, expected_min, {1e-8, 1e-8, 1e-6});
}

void test_ackley(minimizerPtr &solver) {
  VecFun<autodiff::VectorXvar, autodiff::var> f_ad = [](autodiff::VectorXvar v) {
    const int n = static_cast<int>(v.size());
    autodiff::var sum1 = 0.0;
    autodiff::var sum2 = 0.0;

    for (int i = 0; i < n; ++i) {
      sum1 += v(i) * v(i);
      sum2 += cos(2.0 * M_PI * v(i));
    }

    autodiff::var term1 = -20.0 * exp(-0.2 * sqrt(sum1 / n));
    autodiff::var term2 = -exp(sum2 / n);
    return term1 + term2 + 20.0 + std::exp(1.0);
  };

  const int n = 3;
  Vec v(n);
  v << 0.7, -0.6, 0.2;

  Mat m = Mat::Identity(n, n);

  solver->setMaxIterations(10000);
  solver->setTolerance(1.e-10);
  solver->setInitialHessian(m);
  HessFun<Vec, Mat> hfun = [&](const Vec &x_in) { return autodiff_hessian(f_ad, x_in); };
  solver->setHessian(hfun);

  Vec result = solver->solve(v, f_ad);
  Vec g = autodiff_grad(f_ad, result);
  double f0 = autodiff_val(f_ad, v);
  double f1 = autodiff_val(f_ad, result);
  double g0 = autodiff_grad(f_ad, v).norm();
  double g1 = g.norm();

  Vec expected_min = Vec::Zero(n);
  double dist = (result - expected_min).norm();
  Tests::TestSuite<Vec, Mat>::printStatus(f0, f1, g0, g1, dist, expected_min, {1e-3, 1e-6, 1e-2});
}

int main() {
  minimizerPtr bfgs = std::make_shared<BFGS<Vec, Mat>>();
  minimizerPtr lbfgs = std::make_shared<LBFGS<Vec, Mat>>();
  auto gd = std::make_shared<GradientDescent<Vec, Mat>>();
  gd->setMaxIterations(20000);
  gd->setTolerance(1.e-10);

  using GMRES_Solver = Eigen::GMRES<Mat>;
  GMRES_Solver solver = GMRES_Solver();
  minimizerPtr newton = std::make_shared<Newton<Vec, Mat>>();

  solver.setTolerance(1.e-12);
  solver.setMaxIterations(10000);
  auto bfgs_gmres = std::make_shared<BFGS<Vec, Mat, GMRES_Solver>>((solver));

  auto suite = Tests::TestSuite<Vec, Mat>();

  suite.addImplementation(bfgs, "BFGS");
  suite.addImplementation(lbfgs, "LBFGS");
  suite.addImplementation(gd, "GD (full batch)");
  suite.addImplementation(bfgs_gmres, "BFGS + GMRES");
  suite.addImplementation(newton, "Newton");

  suite.addTest("rosenbrock function (autodiff)", test_rosenbrock);
  suite.addTest("ackley function (autodiff)", test_ackley);
  suite.addTest("rastrigin function (autodiff)", test_rastrigin);

  suite.runTests();
}
