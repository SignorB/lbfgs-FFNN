cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "LTO (Link Time Optimization) enabled.")
else()
    message(WARNING "LTO not supported: ${output}")
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-depth=32768>
)

find_package(Eigen3 REQUIRED)

set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)


add_subdirectory(lib/autodiff)

add_executable(test_runner tests/main.cpp)
add_executable(test_runner_autodiff tests/autodiff_tests.cpp)
add_executable(test_mnist tests/mnist/main.cpp)

set(CXX_TARGETS test_runner test_runner_autodiff test_mnist)
foreach(TARGET ${CXX_TARGETS})
    target_link_libraries(${TARGET} PRIVATE autodiff::autodiff Eigen3::Eigen)
endforeach()

set(ENZYME_PATH "/usr/src/Enzyme/enzyme/build/Enzyme/ClangEnzyme-15.so")

if(NOT EXISTS "${ENZYME_PATH}")
    message(FATAL_ERROR "ENZYME NOT FOUND :( !!")
endif()
add_executable(test_burgers tests/burgers/test_burgers.cpp)

set_target_properties(test_burgers PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(test_burgers PRIVATE Eigen3::Eigen autodiff::autodiff)


  target_compile_options(test_burgers PRIVATE 
    -O3
    -march=native          # <--- FONDAMENTALE: Usa AVX/AVX2/AVX512 della tua CPU
    -ffast-math            # <--- MOLTO IMPORTANTE: Velocizza tanh, exp e calcoli floating point
    -fno-math-errno        # <--- Evita controlli inutili sugli errori matematici
    -DNDEBUG               # <--- CRUCIALE PER EIGEN: Disabilita assert e controlli di range
    "SHELL:-Xclang -load -Xclang ${ENZYME_PATH}"
)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    add_executable(test_mnist_cuda tests/cuda/mnist.cu)
    add_executable(test_gen_cuda tests/cuda/synthetic.cu)
    add_executable(test_fashion_cuda tests/cuda/fashion.cu)
    target_link_libraries(test_mnist_cuda PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::cublas CUDA::cudart)
    target_link_libraries(test_gen_cuda PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::cublas CUDA::cudart)
    target_link_libraries(test_fashion_cuda PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::cublas CUDA::cudart)

    message(STATUS "CUDA compiler found. Enabled cuda tests.")
else()
    message(WARNING "CUDA compiler not found. Skipping .cu sources.")
endif()

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling Eigen parallelization.")
    target_link_libraries(test_mnist PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(test_mnist PRIVATE EIGEN_USE_OPENMP)
else()
    message(WARNING "OpenMP not found. Eigen will run in single-thread mode.")
endif()
