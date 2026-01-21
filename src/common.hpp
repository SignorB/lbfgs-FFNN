#pragma once

/**
 * @file common.hpp
 * @brief Common aliases and utilities shared across CPU components.
 */

#include <Eigen/Eigen>
#include <functional>
#include <iostream>

#ifndef NDEBUG
  /// @brief Debug assertion with message and source location.
  #define check(condition, message)                                                                                         \
    do {                                                                                                                    \
      if (!condition) {                                                                                                     \
        std::cerr << "[FAILED ASSERTION]" << std::endl;                                                                     \
        std::cerr << "  Condition: " << #condition << std::endl;                                                            \
        std::cerr << "  Message: " << (message) << std::endl;                                                               \
        std::cerr << "  File: " << __FILE__ << ", Line: " << __LINE__ << std::endl;                                         \
        std::cerr << "  Aborting..." << std::endl;                                                                          \
        std::abort();                                                                                                       \
      }                                                                                                                     \
    } while (0)

#else
  /// @brief No-op assertion in release builds.
  #define check(condition, message) ((void)0)
#endif

/// @brief Gradient function type alias (T -> T).
template <typename T> using GradFun = std::function<T(T)>;

/// @brief Objective function type alias (T -> W).
template <typename T, typename W> using VecFun = std::function<W(T)>;

/// @brief Hessian function type alias (V -> M).
template <typename V, typename M> using HessFun = std::function<M(V)>;

#ifdef _OPENMP
  #include <omp.h>
#endif

/// @brief Print Eigen/OpenMP parallelism settings for diagnostics.
inline void checkParallelism() {
  std::cout << "=== CHECK EIGEN PARALLELISM ===" << std::endl;

#ifdef EIGEN_USE_OPENMP
  std::cout << "[COMPILE] EIGEN_USE_OPENMP macro: ACTIVE (OK)" << std::endl;
#else
  std::cout << "[COMPILE] EIGEN_USE_OPENMP macro: INACTIVE (Warning!)" << std::endl;
#endif

#ifdef _OPENMP
  std::cout << "[OPENMP] OpenMP detected." << std::endl;
  std::cout << "[OPENMP] Max available threads: " << omp_get_max_threads() << std::endl;
#else
  std::cout << "[OPENMP] OpenMP NOT detected by compiler." << std::endl;
#endif

  std::cout << "[EIGEN] Eigen will use: " << Eigen::nbThreads() << " threads." << std::endl;

  std::cout << "================================" << std::endl << std::endl;
}
