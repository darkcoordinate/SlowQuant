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

#define EMPTY_IDX -128

class op_point{
    public:
    uint16_t hash;
    signed char idx;
    bool anni;

    op_point(signed char idx, bool anni){
        this->idx = idx;
        this->anni = anni;
        hash = anni<<8 | idx;
    }

    // Must define equality for use in hash containers
    bool operator==(const op_point& other) const {
        return hash == other.hash;
    }

    bool operator!=(const op_point& other) const {
        return !(*this == other);
    }
};

struct set{
    op_point key[16];
    int size = -1;
    double value;

    __device__ bool is_empty(){
        return size == -1;
    }

    __device__ bool key_eq(const set& other) const {
        if(size != other.size) return false;
        for(int i = 0; i < size; i++){
            if(key[i] != other.key[i]) return false;
        }    
        return true;
    }

    __device__ size_t hash_set(){
        size_t hash = 0;
        for(int i = 0; i < size; i++){
            hash += key[i].hash;
        }
        return hash;
    }
};

struct FermionicOperator{    // this is fermionic expression.
    set* sets;
    int size;
    int capacity;

    __device__ __host__ size_t hash_func(size_t hash, int capacity){ // this should give a unique hash for each set.
        return hash % capacity;
    }
    __device__ int find(const set& s){
        unsigned int slot = hash_func(hash_set(s), capacity);
        while (sets[slot].is_empty()) {
            if (sets[slot].key_eq(s)) {
                return slot;
            }
            slot = (slot + 1) % capacity;
        }
        return -1; // Not found
    }

    __device__ double get_value(const set& s){
        int slot = find(s);
        if(slot != -1){
            return sets[slot].value;
        }
        return 0.0;
    }

    __device__ void insert(const set& s){
        int slot = find(s);
        if(slot != -1){
            sets[slot].value += s.value;
        }else{
            slot = hash_func(hash_set(s), capacity);
            while (!sets[slot].is_empty()) {
                slot = (slot + 1) % capacity;
            }
            sets[slot] = s;
        }
    }   

    __device__ void insert_atomic(const set& s){
        unsigned int slot = hash_func(hash_set(s), capacity);
        while (sets[slot].is_empty()) {
            if (sets[slot].key_eq(s)) {
                atomicAdd(&sets[slot].value, s.value);
                return;
            }
            slot = (slot + 1) % capacity;
        }
        sets[slot] = s;
    }

    __device__ double operator[](const set& s){
        return get_value(s);
    }

    __device__ void operator+=(const set& s){
        insert(s);
    }

    __device__ void operator-=(const set& s){
        insert(set(s.key, -s.value));
    }

    __device__ void operator*=(const double factor){
        for(int i = 0; i < size; i++){
            sets[i].value *= factor;
        }
    }

    __device__ void operator*=(const FermionicOperator& other){
        FermionicOperator result;
        for(int i = 0; i < size; i++){
            for(int j = 0; j < other.size; j++){
                result.insert(set(sets[i].key + other.sets[j].key, sets[i].value * other.sets[j].value));
            }
        }
        *this = result;
    }   
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

template <typename T, int TABLE_SIZE>
__global__ void scale_points_kernel(Point<T>* in, Point<T>* out, T factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].x = in[idx].x * factor;
        out[idx].y = in[idx].y * factor;
    }
}




py::array_t<double> derivative_theta_ket(
    py::EigenDRef<Eigen::MatrixXd> bra, const py::list &py_ops,
    const py::list &py_ops2, py::EigenDRef<Eigen::MatrixXd> ket,
    const py::object &py_ci_info, const py::array_t<double> &py_thetas,
    const py::object &py_wf_struct, py::bool_ py_do_folding,
    py::int_ specific_state) {

  // py::gil_scoped_release release;
  std::vector<FermionicOperator> ops;
  std::vector<FermionicOperator> ops2;
  std::vector<double> gr_list(py_ops.size());
  int specific_state_ = specific_state.cast<int>();
  std::vector<double> thetas = py_thetas.cast<std::vector<double>>();
  CI_Info ci_info(py_ci_info);
  std::cout << "************************ :";
  std::cout << py_ops.size() << std::endl;
  // auto start = std::chrono::steady_clock::now();

  std::vector<FermionicOperator> T_list;
  for (size_t i = 0; i < py_ops.size(); i++) {
    T_list.push_back(FermionicOperator(
        py_ops[i]
            .attr("operators")
            .cast<std::map<std::vector<std::tuple<int, bool>>, double>>()));
  }
  FermionicOperator Hamiltonian(
      py_ops2[0]
          .attr("operators")
          .cast<std::map<std::vector<std::tuple<int, bool>>, double>>());
  bool do_folding = py_do_folding.cast<bool>();
  std::ofstream MyFile("filename.txt");

#pragma omp parallel for ordered
  for (size_t i = 0; i < py_ops.size(); i++) {
    int tid = omp_get_thread_num();
    double gr = 0;

    // MyFile <<" "<<i<<" " << "\n";
    gr = expectation_vector_SA(bra, {T_list[i], Hamiltonian}, ket, ci_info,
                               thetas, py_wf_struct,
                               do_folding)(specific_state_, specific_state_);
    gr -= expectation_vector_SA(bra, {Hamiltonian, T_list[i]}, ket, ci_info,
                                thetas, py_wf_struct,
                                do_folding)(specific_state_, specific_state_);
    gr_list[i] = gr;
    MyFile << "thread :" << tid << " step :" << i << " " << gr << std::endl;
    // auto end = std::chrono::steady_clock::now();
    // auto diff = end - start;
    // std::cout << " time :" ;
    // std::cout <<
    // std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() <<
    // std::endl; start = end;
  }
  MyFile.close();
  return py::cast(gr_list);
}


PYBIND11_MODULE(fermionic_ops, m) {
    m.def("derivative_theta_ket", &derivative_theta_ket, py::arg("bra"),
        py::arg("op1"), py::arg("op2"), py::arg("ket"), py::arg("ci_info"),
        py::arg("thetas"), py::arg("wf_struct"), py::arg("do_folding") = true,
        py::arg("specific_state"), "good", py::return_value_policy::move);
}