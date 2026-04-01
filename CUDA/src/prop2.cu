#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <omp.h>
#include <unordered_map>
#include <utility>
#include <vector>
#define PYBIND11_BUILD
#ifdef PYBIND11_BUILD
#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif
#include <chrono>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

    __device__ __host__ op_point(signed char idx, bool anni){
        this->idx = idx;
        this->anni = anni;
        hash = (1<<(2*anni))<<8 | idx;
    }

    __device__ __host__ op_point(){
        idx = EMPTY_IDX;
        anni = false;
        hash = 0;
    }
    __device__ __host__ void print() const{
        printf("[idx: %d, anni: %d, hash: %d ]", idx, anni, hash);   
    }

    // Must define equality for use in hash containers
    __device__ __host__ bool operator==(const op_point& other) const {
        return hash == other.hash;
    }

    __device__ __host__ bool operator!=(const op_point& other) const {
        return !(*this == other);
    }
};

struct set{
    op_point key[16];
    int size = -1;
    double value;
    size_t hash;

    __host__ __device__ set(){
        size = -1;
        value = 0.0;
    }

    __host__ __device__ bool is_empty(){
        return size == -1;
    }

    __host__ __device__ bool key_eq(const set& other) const {
        if(size != other.size) return false;
        for(int i = 0; i < size; i++){
            if(key[i] != other.key[i]) return false;
        }    
        return true;
    }

    __host__ __device__ void print(){
        printf("size: %d, value: %f [ ", size, value);
        for(int i = 0; i < size; i++){
            key[i].print();
        }
        printf("]\n");
    }

    __host__ __device__ size_t hash_set() const {
        size_t t_hash = 0;
        for(int i = 0; i < size; i++){
            t_hash +=(key[i].hash <<(8*i));
        }
        this->hash = t_hash;
        return t_hash;
    }
    
};

struct FermionicOperator{    // this is fermionic expression.
    set* sets;
    int size;
    const size_t capacity  = 1<<16;

    __host__ __device__ FermionicOperator(){
        sets = nullptr;
        size = 0;
    }

    __host__ __device__ const  size_t hash_func(const size_t hash, const size_t capacity) const { // this should give a unique hash for each set.
        return hash % capacity;
    }
    __host__ __device__  int find(const set& s){
        const size_t hash = s.hash_set();
        size_t slot = hash_func(hash, capacity);
        if(sets == nullptr){
            return -1;
        }
        while (sets[slot].is_empty()) {
            if (sets[slot].key_eq(s)) {
                return slot;
            }
            slot = (slot + 1) % capacity;
        }
        return -1; // Not found
    }

    __host__ __device__  double get_value(const set& s){
        int slot = find(s);
        if(slot != -1){
            return sets[slot].value;
        }
        return 0.0;
    }

    __host__ __device__  void insert(const set& s){
        int slot = find(s);
        if(slot != -1){
            sets[slot].value += s.value;
        }else{
            slot = hash_func(s.hash_set(), capacity);
            if(sets == nullptr){
                sets = new set[capacity];
            }
            while (!sets[slot].is_empty()) {
                slot = (slot + 1) % capacity;
            }
            sets[slot] = s;
        }
    }

    __host__ __device__ double get_value(const set& s){
        int slot = find(s);
        if(slot != -1){
            return sets[slot].value;
        }
        return 0.0;
    }

    __host__ __device__ double operator[](const set& s){
        return get_value(s);
    }

    // __host__ __device__  void insert_atomic(const set& s){
    //     unsigned int slot = hash_func(s.hash_set(), capacity);
    //     while (sets[slot].is_empty()) {
    //         if (sets[slot].key_eq(s)) {
    //             atomicAdd(&sets[slot].value, s.value);
    //             return;
    //         }
    //         slot = (slot + 1) % capacity;
    //     }
    //     sets[slot] = s;
    // }
};

__host__ __device__ size_t hash_op_point(struct op_point p) {
    // Map signed char to 0-255 range to prevent negative results
    unsigned char val = (unsigned char)p.idx;
    // Combine: [anni (1 bit)][idx (8 bits)]
    return (p.anni ? 256 : 0) + val;
}



__host__ __device__ size_t hash_set(const set& s) {
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

class CI_Info {
public:
  int num_active_elec_alpha;
  int num_active_elec_beta;
  int num_active_orbs;
  int num_inactive_orbs;
  int num_virtual_orbs;
  int space_extension_offset;
  std::map<uint64_t, uint64_t> det2idx;
  std::vector<uint64_t> idx2det;



  CI_Info(py::object py_ci_info) {
    py::dict d = py_ci_info.attr("det2idx").cast<py::dict>();
    for (auto item : d) {
      det2idx[item.first.cast<uint64_t>()] = item.second.cast<uint64_t>();
    }
    idx2det = py_ci_info.attr("idx2det").cast<std::vector<uint64_t>>();
    num_active_elec_alpha =
        py_ci_info.attr("num_active_elec_alpha").cast<int>();
    num_active_elec_beta = py_ci_info.attr("num_active_elec_beta").cast<int>();
    num_active_orbs = py_ci_info.attr("num_active_orbs").cast<int>();
    num_inactive_orbs = py_ci_info.attr("num_inactive_orbs").cast<int>();
    num_virtual_orbs = py_ci_info.attr("num_virtual_orbs").cast<int>();
    space_extension_offset =
        py_ci_info.attr("space_extension_offset").cast<int>();
  }
};



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

  //std::vector<FermionicOperator> T_list;
  for (size_t i = 0; i < py_ops.size(); i++) {
    //std::cout << py_ops[i].attr("operators").attr("keys") << std::endl;
    FermionicOperator op;
    py::dict d = py_ops[i].attr("operators").cast<py::dict>();
    for (auto item : d) {
        py::tuple t = item.first.cast<py::tuple>();
        set s;
        s.size = t.size();
        for (size_t j = 0; j < t.size(); j++) {
            py::tuple t2 = t[j].cast<py::tuple>();
            std::cout <<"T_list"<<i<<": "<<t2[0].cast<int>() << " " << t2[1].cast<bool>() << std::endl;
            s.key[j] = op_point{t2[0].cast<signed char>(), t2[1].cast<bool>()};
            std::cout<<"Happe"<<std::endl;
        }
        std::cout<<"Naaa"<<item.second<<std::endl;
        s.value = item.second.cast<double>();
        s.print();
        op.insert(s);
    }
    //auto c =  py_ops[i].attr("operators").cast<set>();
    //T_list.push_back(FermionicOperator(py_ops[i].cast<>()));
  }
//   FermionicOperator Hamiltonian(
//       py_ops2[0]
//           .attr("operators")
//           .cast<std::map<std::vector<std::tuple<int, bool>>, double>>());
//   bool do_folding = py_do_folding.cast<bool>();
//   std::ofstream MyFile("filename.txt");

// #pragma omp parallel for ordered
//   for (size_t i = 0; i < py_ops.size(); i++) {
//     int tid = omp_get_thread_num();
//     double gr = 0;

//     // MyFile <<" "<<i<<" " << "\n";
//     gr = expectation_vector_SA(bra, {T_list[i], Hamiltonian}, ket, ci_info,
//                                thetas, py_wf_struct,
//                                do_folding)(specific_state_, specific_state_);
//     gr -= expectation_vector_SA(bra, {Hamiltonian, T_list[i]}, ket, ci_info,
//                                 thetas, py_wf_struct,
//                                 do_folding)(specific_state_, specific_state_);
//     gr_list[i] = gr;
//     MyFile << "thread :" << tid << " step :" << i << " " << gr << std::endl;
//     // auto end = std::chrono::steady_clock::now();
//     // auto diff = end - start;
//     // std::cout << " time :" ;
//     // std::cout <<
//     // std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() <<
//     // std::endl; start = end;
//   }
//   MyFile.close();
  return py::cast(gr_list);
}


PYBIND11_MODULE(fermionic_ops, m) {
    m.def("derivative_theta_ket", &derivative_theta_ket, py::arg("bra"),
        py::arg("op1"), py::arg("op2"), py::arg("ket"), py::arg("ci_info"),
        py::arg("thetas"), py::arg("wf_struct"), py::arg("do_folding") = true,
        py::arg("specific_state"), "good", py::return_value_policy::move);
}