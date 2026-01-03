cmake_minimum_required(VERSION 3.14)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#check compiler optimization also for linker
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "LTO (Link Time Optimization) enabled.")
else()
    message(WARNING "LTO not supported: ${output}")
endif()

add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)

find_package(Eigen3 REQUIRED)

set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)


add_subdirectory(lib/autodiff)

add_executable(main_app src/main.cpp)
add_executable(test_runner tests/main.cpp)
add_executable(test_runner_autodiff tests/autodiff_tests.cpp)
add_executable(test_mnist tests/mnist/main.cpp)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling Eigen parallelization.")
    target_link_libraries(test_mnist PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(test_mnist PRIVATE EIGEN_USE_OPENMP)
else()
    message(WARNING "OpenMP not found. Eigen will run in single-thread mode.")
endif()

target_link_libraries(test_mnist PRIVATE autodiff::autodiff Eigen3::Eigen)
target_link_libraries(main_app PRIVATE autodiff::autodiff Eigen3::Eigen)
target_link_libraries(test_runner PRIVATE autodiff::autodiff Eigen3::Eigen)
target_link_libraries(test_runner_autodiff PRIVATE autodiff::autodiff Eigen3::Eigen)

