cmake_minimum_required(VERSION 3.18)
project(fermionic_ops_cuda LANGUAGES CXX CUDA)

# 1. Find Dependencies
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "-Xcompiler -fPIC")

set(CMAKE_CXX_COMPILER "nvcc")
set(CMAKE_C_COMPILER "nvcc")

# 2. Define the Python Module
# The module name (solver_accel) must match the PYBIND11_MODULE name in C++

pybind11_add_module(fermionic_ops_cuda  src/m4.cu src/prop.cu)

# 3. Link Libraries
target_link_libraries(fermionic_ops_cuda PRIVATE 
    Eigen3::Eigen 
    pybind11::module
)

# 4. CUDA Specifics (Important for your kernels)
set_target_properties(fermionic_ops_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)