#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}




namespace py = pybind11;

// Forward declaration of the CUDA launcher
void launch_vector_add(const float* a, const float* b, float* c, int n);

py::array_t<float> add_arrays(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request(), buf_b = b.request();
    int n = buf_a.size;

    auto result = py::array_t<float>(n);
    auto buf_res = result.request();

    launch_vector_add((float*)buf_a.ptr, (float*)buf_b.ptr, (float*)buf_res.ptr, n);

    return result;
}

PYBIND11_MODULE(cuda_module, m) {
    m.def("add_arrays", &add_arrays, "Add two arrays on GPU");
}