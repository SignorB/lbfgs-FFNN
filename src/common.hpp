#pragma once

#include <Eigen/Eigen>
#include <functional>
#include <iostream>

#ifndef NDEBUG
  #define check(condition, message)                                   \
    do {                                                              \
      if (!condition) {                                               \
        std::cerr << "[FAILED ASSERTION]" << std::endl;               \
        std::cerr << "  Condition: " << #condition << std::endl;      \
        std::cerr << "  Message: " << (message) << std::endl;         \
        std::cerr << "  File: " << __FILE__ << ", Line: " << __LINE__ \
                  << std::endl;                                       \
        std::cerr << "  Aborting..." << std::endl;                    \
        std::abort();                                                 \
      }                                                               \
    } while (0)

#else
  #define check(condition, message) ((void)0)
#endif

template <typename T>
using GradFun = std::function<T(T)>;

template <typename T, typename W>
using VecFun = std::function<W(T)>;

template <typename V, typename M>
using HessFun = std::function<M(V)>;



#ifdef _OPENMP
#include <omp.h>
#endif

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
