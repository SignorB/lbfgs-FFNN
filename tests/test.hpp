#include "../src/minimizer/full_batch_minimizer.hpp"
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace Tests {

/**
 * @brief Generic test suite for minimizer implementations.
 *
 * This class allows you to:
 *  - register multiple minimizer implementations (e.g. BFGS, L-BFGS, ...)
 *  - register multiple tests (each test being a function using a minimizer)
 *  - run all tests on all implementations, collecting timing and iteration info.
 *
 * @tparam V Vector type used by the minimizers (e.g. Eigen::VectorXd).
 * @tparam M Matrix type used by the minimizers (e.g. Eigen::MatrixXd).
 */
template <typename V, typename M>
class TestSuite {
  /// Shared pointer to a generic minimizer implementation.
  using minimizerPtr = std::shared_ptr<cpu_mlp::FullBatchMinimizer<V, M>>;
  /// Type of a test function: takes a minimizer instance by reference.
  using testFunction = std::function<void(minimizerPtr &)>;

public:
  struct StatusConfig {
    double f_tol = 1e-6;
    double g_tol = 1e-6;
    double dist_tol = 1e-4;
  };

  /**
   * @brief Utility to summarize convergence against a known target.
   *
   * Prints initial/final loss and gradient norms, distance to the target, and
   * classifies the outcome (global minimum within tolerances, stationary away
   * from target, or not converged).
   */
  static void printStatus(double f0,
                          double f1,
                          double g0,
                          double g1,
                          double dist,
                          const V &expected,
                          StatusConfig cfg = default_status_cfg) {
    std::string status;
    if (f1 <= cfg.f_tol && g1 <= cfg.g_tol && dist <= cfg.dist_tol)
      status = "global minimum";
    else if (g1 <= cfg.g_tol && dist > cfg.dist_tol)
      status = "stationary point";
    else
      status = "not converged";

    std::cout << " f0=" << f0 << " f1=" << f1
              << " ||g0||=" << g0 << " ||g1||=" << g1
              << " dist_to_target=" << dist << " -> " << status << std::endl;

    if (dist > cfg.dist_tol)
      std::cout << "    target: " << expected.transpose() << std::endl;
  }

  /// Override default tolerances used by printStatus when no per-call override is provided.
  static void setDefaultStatusConfig(const StatusConfig &cfg) {
    default_status_cfg = cfg;
  }

  /// Get current default tolerances.
  static StatusConfig defaultStatusConfig() { return default_status_cfg; }
  /**
   * @brief Construct an empty test suite.
   *
   * Initializes the internal containers used to store implementations
   * and tests.
   */
  TestSuite() {
    impls = std::map<std::string, minimizerPtr>();
    tests = std::vector<std::pair<std::string, testFunction>>();
  }

  /**
   * @brief Register a minimizer implementation in the suite.
   *
   * The implementation is associated with a human-readable name and will
   * be used for all registered tests when @ref runTests is called.
   *
   * @param ptr  Shared pointer to a minimizer instance.
   * @param name Identifier for this implementation (used in output).
   */
  void addImplementation(minimizerPtr ptr, std::string name) {
    impls[name] = ptr;
  }

  /**
   * @brief Register a test in the suite.
   *
   * A test is a callable object that receives a reference to a minimizer
   * implementation and typically:
   *  - sets up the optimization problem,
   *  - calls solve(...),
   *  - checks/prints results.
   *
   * @param name Descriptive name of the test (used in output).
   * @param fun  Test function to be executed on each implementation.
   */
  void addTest(std::string name, testFunction fun) {
    tests.push_back(std::make_pair(name, fun));
  }

  /**
   * @brief Run all registered tests on all registered implementations.
   *
   * For each test, this method:
   *  - prints a header with the test name,
   *  - runs the test on every registered implementation,
   *  - measures wall-clock time using std::chrono::steady_clock,
   *  - prints elapsed time, number of iterations, and tolerance used
   *    by the minimizer.
   */
  void runTests() {
    for (std::pair<std::string, testFunction> &test : tests) {
      std::cout << "======================"
                   "RUNNING TEST:"
                << test.first << "======================" << std::endl;

      for (auto &impl : impls) {
        std::cout << "  implementation: " << impl.first << std::endl;

        auto before = std::chrono::steady_clock::now();

        // Execute the test on the current implementation
        test.second(impl.second);

        auto after = std::chrono::steady_clock::now();
        auto delta_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after - before)
                .count();

        std::cout << "\t time elapsed: " << delta_us << " us" << std::endl;
        std::cout << "\t iterations:   " << impl.second->iterations()
                  << std::endl;
        std::cout << "\t tolerance:    " << impl.second->tolerance()
                  << std::endl;
      }
    }
  }

private:
  /// Map from implementation name to minimizer instance.
  std::map<std::string, minimizerPtr> impls;

  /// List of (test name, test function) pairs.
  std::vector<std::pair<std::string, testFunction>> tests;

  inline static StatusConfig default_status_cfg{};
};

} // namespace Tests
