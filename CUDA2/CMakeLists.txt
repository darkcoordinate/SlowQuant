cmake_minimum_required(VERSION 3.18)
project(cuco_dynamic_map_example LANGUAGES CXX CUDA)
#set(CUDA_PATH "/usr/local/cuda-12.9")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT /usr/local/cuda)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable exceptions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -exceptions")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -exceptions")

set(CUDAToolkit_ROOT "/usr/local/cuda")
#find_package(CUDA 12.9 REQUIRED)
find_package(CUDAToolkit REQUIRED)
# Find required packages

#find_package(cccl REQUIRED)  # Requires cuCollections installed
message(STATUS "CUDA Toolkit found!")

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")
# Create executable
add_executable(dynamic_map_example src/m9.cu)

# Link libraries
target_link_libraries(dynamic_map_example
    PRIVATE
#    cuco::cuco
    CUDA::cudart
)

target_include_directories(dynamic_map_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extra/cuCollections/include
${CUDAToolkit_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/extra/cccl/
${CMAKE_CURRENT_SOURCE_DIR}/extra/cccl/thrust
${CMAKE_CURRENT_SOURCE_DIR}/extra/cccl/libcudacxx/include 
${CMAKE_CURRENT_SOURCE_DIR}/extra/cccl/cub
)
# Set properties
set_target_properties(dynamic_map_example
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)