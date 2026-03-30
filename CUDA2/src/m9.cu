#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>

// 1. A template struct (similar to a cuco 'Slot')
template <typename T>
struct Point {
    T x;
    T y;
};

struct op_point{
    signed char idx;
    bool anni;
    // Must define equality for use in hash containers
    bool operator==(const op_point& other) const {
        return idx == other.idx && anni == other.anni;
    }
};

struct set{
    op_point key[16];
    int size;
    double value;
};


__device__ __host__ size_t hash_op_point(struct op_point p) {
    // Map signed char to 0-255 range to prevent negative results
    unsigned char val = (unsigned char)p.idx;
    // Combine: [anni (1 bit)][idx (8 bits)]
    return (p.anni ? 256 : 0) + val;
}



__device__ __host__ size_t hash_set(const set& s) {
    size_t hash = 0x811c9dc5; // FNV offset basis
    for (int i = 0; i < s.size; ++i) {
        // Pack op_point into a single byte + bit (0-511)
        uint16_t combined = (static_cast<uint16_t>(s.key[i].anni) << 8) | 
                            (static_cast<uint8_t>(s.key[i].idx));
        
        // FNV-1a XOR-multiply step
        hash ^= combined;
        hash *= 0x01000193; // FNV prime
    }
    return hash;
}



template <typename T>
class CudaDictionary {
public:
    T* d_table;
    int capacity;

    ~CudaDictionary() {
        cudaFree(d_table);
    }
};

__device__ unsigned int hash_func(unsigned int key, int capacity) {
    // Simple modulo hash; for production, use MurmurHash or similar
    return key % capacity;
}

template <typename T>
__device__ void insert_kernel(T* table, unsigned int key, int value, int capacity) {
    unsigned int slot = hash_func(key, capacity);

    while (true) {
        // Try to claim the slot if it's empty
        unsigned int prev = atomicCAS(&table[slot].key, EMPTY_KEY, key);

        if (prev == EMPTY_KEY || prev == key) {
            table[slot].value = value;
            return;
        }

        // Linear probing: move to the next slot on collision
        slot = (slot + 1) % capacity;
    }
}

__device__ int search_dict(DictEntry* table, unsigned int key, int capacity) {
    unsigned int slot = hash_func(key, capacity);

    while (table[slot].key != EMPTY_KEY) {
        if (table[slot].key == key) {
            return table[slot].value;
        }
        slot = (slot + 1) % capacity;
    }
    return -1; // Not found
}


// 2. A Template Kernel with __restrict__ optimization
// BLOCK_SIZE is a template constant for loop unrolling
template <typename T, int BLOCK_SIZE>
__global__ void scale_points_kernel(
    const Point<T>* __restrict__ input, 
    Point<T>*       __restrict__ output, 
    T factor, 
    int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // High-performance: compiler knows 'input' and 'output' don't alias
        T new_x = input[idx].x * factor;
        T new_y = input[idx].y * factor;

        output[idx] = Point<T>{new_x, new_y};
    }
}

int main() {
    const int N = 1024;
    const float factor = 2.0f;

    // 3. Prepare Data using Thrust
    thrust::host_vector<Point<float>> h_in(N);
    for(int i = 0; i < N; ++i) {
        h_in[i] = { (float)i, (float)i * 10.0f };
    }

    thrust::device_vector<Point<float>> d_in = h_in;
    thrust::device_vector<Point<float>> d_out(N);

    // 4. Launch Template Kernel
    // We specify <float, 256> to the template
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    scale_points_kernel<float, 256><<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_in.data()), 
        thrust::raw_pointer_cast(d_out.data()), 
        factor, 
        N
    );

    // 5. Verification
    thrust::host_vector<Point<float>> h_out = d_out;
    std::cout << "Input[5]: (" << h_in[5].x << ", " << h_in[5].y << ")\n";
    std::cout << "Scaled[5]: (" << h_out[5].x << ", " << h_out[5].y << ")\n";

    return 0;
}
