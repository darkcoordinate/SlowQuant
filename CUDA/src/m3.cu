#define EIGEN_USE_GPU 1
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <chrono>
// CUDA kernel for 6x6 matrix multiplication
__global__ void matrixMul6x6(Eigen::Matrix<float, 6, 6>* output, 
                              const Eigen::Matrix<float, 6, 6>* A, 
                              const Eigen::Matrix<float, 6, 6>* B,
                              int numMatrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numMatrices) {
        // Get the matrices for this index
        Eigen::Matrix<float, 6, 6> a = A[idx];
        Eigen::Matrix<float, 6, 6> b = B[idx];
        Eigen::Matrix<float, 6, 6> c;
        
        // Perform matrix multiplication
        // C = A * B
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float sum = 0.0f;
                for (int k = 0; k < 6; k++) {
                    sum += a(i, k) * b(k, j);
                }
                c(i, j) = sum;
            }
        }
        
        // Store result
        output[idx] = c;
    }
}

// Optimized kernel using Eigen's internal operations
__global__ void matrixMul6x6Optimized(Eigen::Matrix<float, 6, 6>* output, 
                                       const Eigen::Matrix<float, 6, 6>* A, 
                                       const Eigen::Matrix<float, 6, 6>* B,
                                       int numMatrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numMatrices) {
        // Use Eigen's built-in multiplication (faster)
        output[idx] = A[idx] * B[idx];
    }
}


// Helper function to print a 6x6 matrix
void printMatrix(const Eigen::Matrix<float, 6, 6>& mat, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {

  int device;
    cudaGetDeviceCount(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor (SM): " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    
    // Total potential concurrent threads on the whole GPU:
    int total_threads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    std::cout << "Total simultaneous threads: " << total_threads << std::endl;


    int minGridSize;
    int blockSize;

    // This calculates the best block size for your specific kernel function  
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMul6x6Optimized, 0, 0);

    std::cout << "Suggested Block Size: " << blockSize << std::endl;


    std::cout << "========================================" << std::endl;
    std::cout << "6x6 Matrix Multiplication with CUDA + Eigen" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Number of matrix pairs to multiply
    int numMatrices = 100;
    
    // Allocate host memory for matrices
    Eigen::Matrix<float, 6, 6>* h_A = new Eigen::Matrix<float, 6, 6>[numMatrices];
    Eigen::Matrix<float, 6, 6>* h_B = new Eigen::Matrix<float, 6, 6>[numMatrices];
    Eigen::Matrix<float, 6, 6>* h_C = new Eigen::Matrix<float, 6, 6>[numMatrices];
    
    // Initialize matrices with random values
    std::cout << "Initializing " << numMatrices << " pairs of 6x6 matrices..." << std::endl;
    for (int i = 0; i < numMatrices; i++) {
        h_A[i] = Eigen::Matrix<float, 6, 6>::Random();
        h_B[i] = Eigen::Matrix<float, 6, 6>::Random();
    }
    
    // Print first matrix as example
    std::cout << "\nExample matrices (first pair):" << std::endl;
    printMatrix(h_A[0], "Matrix A[0]");
    printMatrix(h_B[0], "Matrix B[0]");
    
    // Allocate device memory
    Eigen::Matrix<float, 6, 6>* d_A;
    Eigen::Matrix<float, 6, 6>* d_B;
    Eigen::Matrix<float, 6, 6>* d_C;
    
    size_t matrixSize = sizeof(Eigen::Matrix<float, 6, 6>);
    size_t totalSize = matrixSize * numMatrices;
    
    cudaError_t err;
    
    err = cudaMalloc(&d_A, totalSize);
    err = cudaMalloc(&d_B, totalSize);
    err = cudaMalloc(&d_C, totalSize);
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy data to device
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(d_A, h_A, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numMatrices + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "Launching CUDA kernel with " << blocksPerGrid << " blocks, "
              << threadsPerBlock << " threads/block..." << std::endl;
    
    // Use optimized kernel with Eigen's built-in multiplication
    matrixMul6x6Optimized<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, numMatrices);
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back to host
    std::cout << "Copying results back to CPU..." << std::endl;
    cudaMemcpy(h_C, d_C, totalSize, cudaMemcpyDeviceToHost);
    
    // Verify first result with CPU computation
    std::cout << "\nVerifying results..." << std::endl;
    Eigen::Matrix<float, 6, 6> cpu_result = h_A[0] * h_B[0];
    Eigen::Matrix<float, 6, 6> gpu_result = h_C[0];
    
    float error = (cpu_result - gpu_result).norm();
    
    std::cout << "\nFirst matrix multiplication result:" << std::endl;
    printMatrix(gpu_result, "GPU Result");
    printMatrix(cpu_result, "CPU Result (verification)");
    
    std::cout << "Error norm: " << error << std::endl;
    
    if (error < 1e-5) {
        std::cout << "✓ SUCCESS! All " << numMatrices << " matrices multiplied correctly!" << std::endl;
    } else {
        std::cout << "✗ ERROR: Results don't match!" << std::endl;
    }
    
    // Benchmark
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Warm-up
    matrixMul6x6Optimized<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, numMatrices);
    cudaDeviceSynchronize();
    
    // Time GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMul6x6Optimized<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, numMatrices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    
    // Time CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numMatrices; i++) {
        Eigen::Matrix<float, 6, 6> result = h_A[i] * h_B[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    std::cout << "GPU time: " << gpuTime << " ms for " << numMatrices << " matrices" << std::endl;
    std::cout << "CPU time: " << cpuTime << " ms for " << numMatrices << " matrices" << std::endl;
    std::cout << "Speedup: " << (cpuTime / gpuTime) << "x" << std::endl;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
}