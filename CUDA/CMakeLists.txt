cmake_minimum_required(VERSION 3.18)
project(QuantumSolver LANGUAGES CXX CUDA)

# 1. Find Dependencies
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC")

set(CMAKE_CXX_COMPILER "nvcc")
set(CMAKE_C_COMPILER "nvcc")

# 2. Define the Python Module
# The module name (solver_accel) must match the PYBIND11_MODULE name in C++

pybind11_add_module(solver_accel src/kernels.cu)

# 3. Link Libraries
target_link_libraries(solver_accel PRIVATE 
    Eigen3::Eigen 
    pybind11::module
)

# 4. CUDA Specifics (Important for your kernels)
set_target_properties(solver_accel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)