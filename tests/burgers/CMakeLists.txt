cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/clang++-15")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++-15" CACHE FILEPATH "C++ compiler" FORCE)
endif()

project(BurgersParallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(ENZYME_PLUGIN_PATH "/usr/src/Enzyme/enzyme/build/Enzyme/ClangEnzyme-15.so" CACHE FILEPATH "Path to Clang Enzyme plugin")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(WARNING "Burgers parallel build expects Clang; current compiler is ${CMAKE_CXX_COMPILER_ID}.")
endif()
if(ENZYME_PLUGIN_PATH STREQUAL "")
    message(WARNING "ENZYME_PLUGIN_PATH is empty; pass -DENZYME_PLUGIN_PATH=/path/to/ClangEnzyme-XX.so")
elseif(NOT EXISTS "${ENZYME_PLUGIN_PATH}")
    message(WARNING "ENZYME_PLUGIN_PATH not found: ${ENZYME_PLUGIN_PATH}")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(test_burgers_parallel "${PROJECT_ROOT}/tests/burgers/test_burgers_parallel.cpp")
target_include_directories(test_burgers_parallel PRIVATE
    "${PROJECT_ROOT}/lib/autodiff"
)
target_link_libraries(test_burgers_parallel PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(test_burgers_parallel PRIVATE -O3 -fopenmp -fpass-plugin=${ENZYME_PLUGIN_PATH})
target_compile_definitions(test_burgers_parallel PRIVATE EIGEN_USE_OPENMP)
