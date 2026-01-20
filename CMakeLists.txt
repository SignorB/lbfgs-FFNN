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
    $<$<COMPILE_LANGUAGE:CXX>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-DNDEBUG>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

find_package(Eigen3 REQUIRED)

set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)


add_subdirectory(lib/autodiff)

add_executable(test_runner tests/main.cpp)
add_executable(test_runner_autodiff tests/autodiff_tests.cpp)

set(CXX_TARGETS test_runner test_runner_autodiff)
foreach(TARGET ${CXX_TARGETS})
    target_link_libraries(${TARGET} PRIVATE autodiff::autodiff Eigen3::Eigen)
endforeach()




option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
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

        set_source_files_properties(tests/mnist/main.cpp PROPERTIES LANGUAGE CUDA)

        add_executable(test_mnist_cuda tests/cuda/mnist.cu)
        add_executable(test_fashion_cuda tests/cuda/fashion.cu)

        
        target_link_libraries(test_mnist_cuda PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::cublas CUDA::cudart)
        target_link_libraries(test_fashion_cuda PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::cublas CUDA::cudart)

        message(STATUS "CUDA compiler found. Enabled cuda tests.")
    else()
    message(WARNING "CUDA compiler not found. Skipping .cu sources.")
    endif()
endif()

# Unified MNIST Test (Works on CPU too)
add_executable(test_mnist tests/mnist/main.cpp)
target_link_libraries(test_mnist PRIVATE autodiff::autodiff Eigen3::Eigen)

if(ENABLE_CUDA)
    target_link_libraries(test_mnist PRIVATE CUDA::cublas CUDA::cudart)
endif()

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling Eigen parallelization.")
    target_link_libraries(test_mnist PRIVATE OpenMP::OpenMP_CXX)
    
    if(TARGET test_mnist)
        target_link_libraries(test_mnist PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test_mnist PRIVATE EIGEN_USE_OPENMP)
    endif()

    target_compile_definitions(test_mnist PRIVATE EIGEN_USE_OPENMP)
    
    if(ENABLE_CUDA AND TARGET test_mnist)
        target_compile_options(test_mnist PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
    endif()
else()
    message(WARNING "OpenMP not found. Eigen will run in single-thread mode.")
endif()

