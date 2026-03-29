
#include <tuple>
#define EIGEN_USE_GPU 1
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
#include <Eigen/Core>
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

__device__ __host__ static inline int bitcount(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}

#define MAX_OP_LEN 10


struct operators {
  int creator[MAX_OP_LEN];
  int annihilator[MAX_OP_LEN];
  double factor;
  int len_c;
  int len_a;
  int len;

  void print() const {
    std::cout << "Creator ";
    for (size_t i = 0; i < len_c; i++) {
      std::cout << creator[i] << " ";
    }
    std::cout << "Annhilator ";
    for (size_t i = 0; i < len_a; i++) {
      std::cout << annihilator[i] << " ";
    }
    std::cout << factor << " \n";
  }

  void print_c() const{
    printf("Creator ");
    for (size_t i = 0; i < len_c; i++) {
      printf("%d ", creator[i]);
    }
    printf("Annhilator ");
    for (size_t i = 0; i < len_a; i++) {
      printf("%d ", annihilator[i]);
    }
    printf("%f \n", factor);
  }

  __host__ __device__ operators(){
    len_c = 0;
    len_a = 0;
    factor = 0.0;
    len = 0;
  }
} ;

__device__ __host__  void apply_operator_SA_c(const Eigen::MatrixXd &state,
                                    const std::vector<uint64_t> &idx2det,
                                    const std::map<uint64_t, uint64_t> &det2idx,
                                    const uint64_t det_lookup_size,
                                    const int n_dets, const operators &ops,
                                    const int num_active_orbs,
                                    const std::vector<uint64_t> &parity_check,
                                  Eigen::MatrixXd &tmp_state2
                                  ) {

  //Eigen::MatrixXd tmp_state2 =
  //    Eigen::MatrixXd::Zero(state.rows(), state.cols());
  for (int i = 0; i < n_dets; ++i) {
    bool is_non_zero = (state.col(i).array().abs() > 1e-14).any();
    if (!is_non_zero)
      continue;
    uint64_t det = idx2det[i];
    int phase_changes = 0;
    int killstate = 0;

    /* ---- Apply annihilation operators ---- */

    for (int a = static_cast<int>(ops.len_a) - 1; a >= 0; --a) {
      int orb_idx = ops.annihilator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 0) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;

    /* ---- Apply creation operators ---- */
    for (int a = static_cast<int>(ops.len_c) - 1; a >= 0; --a) {
      int orb_idx = ops.creator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 1) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;
    int new_idx = det2idx.at(static_cast<int>(det));
    double sign = (phase_changes % 2 == 0) ? 1.0 : -1.0;
    tmp_state2.col(new_idx) += ops.factor * sign * state.col(i);
  }
}


// __global__ void loop(const Eigen::Matrix<double,Dynamic,> &state,
//                                     // const std::vector<uint64_t> &idx2det,
//                                     // const std::map<uint64_t, uint64_t> &det2idx,
//                                     // const uint64_t det_lookup_size,
//                                     // const int n_dets, const operators* ops,
//                                     // const int num_active_orbs,
//                                     // const std::vector<uint64_t> &parity_check,
//                                     // double ** tmp_stateV,
//                                     // int rows, int cols
//                                   ){
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;

//   //Eigen::MatrixXd state_d = Eigen::Map<const Eigen::MatrixXd>(state, rows, cols);
//   // Eigen::MatrixXd tmp_stateV_d = Eigen::Map<Eigen::MatrixXd>(tmp_stateV[idx], rows, cols);
//   // apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
//   //                           ops[idx], num_active_orbs, parity_check,tmp_stateV_d);
// }
#define EMPTY_KEY 0xffffffff

struct DictEntry {
    unsigned int key;
    int value;
};

class CudaDictionary {
public:
    DictEntry* d_table;
    int capacity;

    // The Constructor: Converts Python Dict to CUDA memory
    CudaDictionary(py::dict source) {
        // 1. Determine size (Load factor: 2x the items for performance)
        int num_items = source.size();
        capacity = num_items * 2; 
        size_t bytes = capacity * sizeof(DictEntry);

        // 2. Allocate and Initialize GPU memory to "Empty" (0xff)
        cudaMalloc(&d_table, bytes);
        cudaMemset(d_table, 0xff, bytes);

        // 3. Prepare data on Host (CPU) first
        std::vector<DictEntry> h_buffer(capacity, {0xffffffff, -1});

        // 4. Iterate through Python Dict and perform Linear Probing on CPU
        for (auto item : source) {
            unsigned int key = item.first.cast<unsigned int>();
            int value = item.second.cast<int>();

            // Simple Linear Probing to find the right slot
            unsigned int slot = key % capacity;
            while (h_buffer[slot].key != 0xffffffff) {
                slot = (slot + 1) % capacity;
            }
            h_buffer[slot] = {key, value};
        }

        // 5. Copy the fully constructed hash table to the GPU
        cudaMemcpy(d_table, h_buffer.data(), bytes, cudaMemcpyHostToDevice);
    }

    ~CudaDictionary() {
        cudaFree(d_table);
    }
};

__device__ unsigned int hash_func(unsigned int key, int capacity) {
    // Simple modulo hash; for production, use MurmurHash or similar
    return key % capacity;
}

__device__ void insert_kernel(DictEntry* table, unsigned int key, int value, int capacity) {
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

__device__ bool  get_col_non_zero(double* state, int col_idx, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        if (state[rows*col_idx + i] > 1e-14) {
            return true;
        }
    }
    return false;
}


__device__ void  assign_col(double* tmp_stateV  , int col_idx, int rows, int cols) {

    for (int i = 0; i < rows; ++i) {
        tmp_stateV[rows*col_idx + i] = 2.0;
    }
}


__device__ void add_col(double * tmp_state, double* state,int new_idx, int col_idx, int rows, int cols, double factor) {
    for (int i = 0; i < rows; ++i) {
        tmp_state[rows*new_idx + i] += state[rows*col_idx + i] * factor;
    }
}


__global__ void loop1(double* state, double* tmp_stateV, int rows, int cols , uint64_t num_ops, operators* ops, int n_dets,
uint64_t* idx2det, uint64_t size_idx2det, CudaDictionary det2idx, int size_det2idx, int num_active_orbs, uint64_t* parity_check
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < n_dets; ++i) {
    //bool is_non_zero = (state.col(i).array().abs() > 1e-14).any();
    bool is_non_zero = get_col_non_zero(state, i, rows, cols);
    if (!is_non_zero)
      continue;
    uint64_t det = idx2det[i];
    int phase_changes = 0;
    int killstate = 0;

    /* ---- Apply annihilation operators ---- */
    for (int a = ops[idx].len_a - 1; a >= 0; --a) {
      int orb_idx = ops[idx].annihilator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 0) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;

    /* ---- Apply creation operators ---- */
    for (int a = ops[idx].len_c - 1; a >= 0; --a) {
      int orb_idx = ops[idx].creator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 1) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;
    //int new_idx = det2idx.at(static_cast<int>(det));
    int new_idx = search_dict(det2idx.d_table, det, det2idx.capacity);
    double sign = (phase_changes % 2 == 0) ? 1.0 : -1.0;
    //tmp_state2.col(new_idx) += ops[idx].factor * sign * state.col(i);
    add_col(tmp_stateV + idx * rows * cols, state, new_idx, i, rows, cols, ops[idx].factor * sign);
  }
}


void loop_test(const Eigen::MatrixXd &state,
                                    const std::vector<uint64_t> &idx2det,
                                    const std::map<uint64_t, uint64_t> &det2idx,
                                    const uint64_t det_lookup_size,
                                    const int n_dets, const operators &ops,
                                    const int num_active_orbs,
                                    const std::vector<uint64_t> &parity_check,
                                  Eigen::MatrixXd &tmp_state2
                                  ){
                                    tmp_state2.col(2).array() += 2.0;
                                  }


__global__ void loop2(double* state, double* tmp_stateV, int rows, int cols , uint64_t num_ops, operators* ops, int n_dets,
uint64_t* idx2det, uint64_t size_idx2det, CudaDictionary det2idx, int size_det2idx, int num_active_orbs, uint64_t* parity_check
){
//assign_col(tmp_stateV + idx * rows * cols, 2, rows, cols);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_ops){
  for (int i = 0; i < n_dets; ++i) {
    //bool is_non_zero = (state.col(i).array().abs() > 1e-14).any();
    //printf("%d %lu %d %d thread %d \n",i, idx2det[i] , ops[idx].len_a, ops[idx].len_c, idx);
    bool is_non_zero = get_col_non_zero(state, i, rows, cols);
    if (!is_non_zero)
      continue;
    
    uint64_t det = idx2det[i];
    
    int phase_changes = 0;
    int killstate = 0;

    /* ---- Apply annihilation operators ---- */
    for (int a = ops[idx].len_a - 1; a >= 0; --a) {
      int orb_idx = ops[idx].annihilator[a];
      //printf("%d %d thread %d ",i, orb_idx , idx);
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 0) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    //printf("\n");
    if (killstate)
      continue;
    
    /* ---- Apply creation operators ---- */
    for (int a = ops[idx].len_c - 1; a >= 0; --a) {
      int orb_idx = ops[idx].creator[a];
      //printf("%d %d thread %d ",i, orb_idx , idx);
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 1) {
        killstate = 1;
        break;
      }
      det ^= mask;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;
    
    //printf("capacity %d \n",det2idx.capacity);
    int new_idx = search_dict(det2idx.d_table, det, det2idx.capacity);
    double sign = (phase_changes % 2 == 0) ? 1.0 : -1.0;
    //printf("new_idx %d\n",new_idx);
    //printf("factor %4.5f \n",ops[idx].factor);
    //p     rintf("sign %4.5f \n",sign);
    //assign_col(tmp_stateV + idx * rows * cols, 2, rows, cols);
    //add_col(tmp_stateV + idx * rows * cols, state, 2, 0, rows, cols, 2.0);  
    add_col(tmp_stateV + idx * rows * cols, state, new_idx, i, rows, cols, ops[idx].factor * sign);

  }
}
}

Eigen::MatrixXd py_opLoop(const py::dict py_ops, const int num_active_orbs,
                          const py::array_t<uint64_t> py_parity_check,
                          const py::array_t<uint64_t> py_idx2det,
                          const py::dict py_det2idx, const bool do_unsafe,
                          const py::EigenDRef<Eigen::MatrixXd> py_state) {
  bool USE_CUDA = true;
  std::vector<uint64_t> idx2det = py_idx2det.cast<std::vector<uint64_t>>();
  std::map<uint64_t, uint64_t> det2idx =
      py_det2idx.cast<std::map<uint64_t, uint64_t>>();
  
  uint64_t det_lookup_size = idx2det.size();
  int n_dets = idx2det.size();
  std::vector<uint64_t> parity_check =
      py_parity_check.cast<std::vector<uint64_t>>();

  CudaDictionary d_det2idx(py_det2idx);
  std::vector<operators> operator2;
  std::vector<operators> operator4;
  std::vector<operators> operator6;
  std::vector<operators> operator8;
  for (auto item : py_ops) {
    py::tuple py_label = item.first.cast<py::tuple>();
    if (py_label.size() == 2) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator[op.len_c++] = orb;
        else
          op.annihilator[op.len_a++] = orb;
      }
      operator2.push_back(op);
    } else if (py_label.size() == 4) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator[op.len_c++] = orb;
        else
          op.annihilator[op.len_a++] = orb;
      }
      operator4.push_back(op);
    }

    else if (py_label.size() == 6) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator[op.len_c++] = orb;
        else
          op.annihilator[op.len_a++] = orb;
      }

      operator6.push_back(op);
    } else if (py_label.size() == 8) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator[op.len_c++] = orb;
        else
          op.annihilator[op.len_a++] = orb;
      }
      operator8.push_back(op);
    } else {
      operators op;
      op.factor = item.second.cast<double>();
      op.len = 0;
      operator2.push_back(op);
    }
  }
  Eigen::MatrixXd state = py_state;
  Eigen::MatrixXd tmp_state = Eigen::MatrixXd::Zero(state.rows(), state.cols());
  std::vector<Eigen::MatrixXd> tmp_stateV(operator2.size() + operator4.size() +
                                          operator6.size() + operator8.size());
  // std::cout << state.format(OctaveFmt) << std::endl;

  // Launch kernel
    int threadsPerBlock = 20;
    int blocksPerGrid = (operator2.size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout<<"block size "<<blocksPerGrid<<" "<<operator2.size()<<std::endl;

    //const size_t matsize = sizeof(state);
    const int rows = (int)state.rows();
    const int cols = (int)state.cols();
    std::cout << "rows " << state.rows() << " cols " << state.cols() << std::endl;
    double* state_device;
    size_t matsize = rows * cols * sizeof(double);
    size_t itc = rows*cols;
    std::cout<<"state  \n"<<state<<std::endl;
    cudaMalloc(&state_device,matsize);
    cudaMemcpy(state_device, state.data(), matsize, cudaMemcpyHostToDevice);


    double* tmp_stateV_device;
    double* tmp_stateV_device2;
    uint64_t* cu_idx2det;
    uint64_t* cu_parity_check;
    operators* cu_operator2;
    operators* cu_operator4;
    operators* cu_operator6;
    operators* cu_operator8;
    cudaMalloc(&cu_operator2,operator2.size()*sizeof(operators));
    cudaMalloc(&cu_operator4,operator4.size()*sizeof(operators));
    cudaMemcpy(cu_operator2,operator2.data(),operator2.size()*sizeof(operators),cudaMemcpyHostToDevice);
    cudaMemcpy(cu_operator4,operator4.data(),operator4.size()*sizeof(operators),cudaMemcpyHostToDevice);
    cudaMalloc(&tmp_stateV_device,matsize*( operator2.size()));
    cudaMalloc(&tmp_stateV_device2,matsize*( operator4.size()));
    
    cudaMalloc(&cu_idx2det,idx2det.size()*sizeof(uint64_t));
    cudaMalloc(&cu_parity_check,parity_check.size()*sizeof(int));
    cudaMemcpy(cu_idx2det,idx2det.data(),idx2det.size()*sizeof(uint64_t),cudaMemcpyHostToDevice);
    cudaMemcpy(cu_parity_check,parity_check.data(),parity_check.size()*sizeof(int),cudaMemcpyHostToDevice);

    loop2<<<blocksPerGrid, threadsPerBlock>>>(state_device, tmp_stateV_device, rows, cols, (uint64_t)operator2.size(),cu_operator2, n_dets, cu_idx2det ,idx2det.size(),d_det2idx,det2idx.size(),num_active_orbs,cu_parity_check);
   
     loop2<<<blocksPerGrid, threadsPerBlock>>>(state_device, tmp_stateV_device2, rows, cols, (uint64_t)operator4.size(),cu_operator4, n_dets, cu_idx2det ,idx2det.size(),d_det2idx,det2idx.size(),num_active_orbs,cu_parity_check);
   
    std::cout << "Kernel launched2" << std::endl;
    
    cudaMemcpy(state.data(), state_device, matsize, cudaMemcpyDeviceToHost);
  
    for(int i = 0; i < operator2.size(); i++){
      tmp_stateV[i] = Eigen::MatrixXd::Zero(state.rows(), state.cols());
      cudaMemcpy(tmp_stateV[i].data(), tmp_stateV_device + (i *itc ), matsize, cudaMemcpyDeviceToHost);
      std::cout << "tmp_stateV[" << i << "] =\n " << tmp_stateV[i] << std::endl;
    }

    for(int i = 0; i < operator4.size(); i++){
      tmp_stateV[i] = Eigen::MatrixXd::Zero(state.rows(), state.cols());
      cudaMemcpy(tmp_stateV[i].data(), tmp_stateV_device2 + (i *itc ), matsize, cudaMemcpyDeviceToHost);
      std::cout << "tmp_stateV[" << i << "] =\n " << tmp_stateV[i] << std::endl;
    }
    

    cudaFree(state_device);
    cudaFree(tmp_stateV_device);
    cudaFree(tmp_stateV_device2);
    cudaFree(cu_idx2det);
    cudaFree(cu_parity_check);
    cudaFree(cu_operator2);
    cudaFree(cu_operator4);
    cudaFree(cu_operator6);
    cudaFree(cu_operator8);
    
    for(int i = 0; i < operator2.size(); i++){
      tmp_stateV[i] = Eigen::MatrixXd::Zero(state.rows(), state.cols());
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator2[i], num_active_orbs, parity_check,tmp_stateV[i]);
      std::cout << "tmp_state cuda[" << i << "] =\n " << tmp_stateV[i] << std::endl;
    }

    for(int i = 0; i < operator4.size(); i++){
      tmp_stateV[i] = Eigen::MatrixXd::Zero(state.rows(), state.cols());
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator4[i], num_active_orbs, parity_check,tmp_stateV[i]);
      std::cout << "tmp_state cuda[" << i << "] =\n " << tmp_stateV[i] << std::endl;
    }
    std::cout << "Kernel launched" << std::endl;
    if(USE_CUDA == false){
      #pragma omp parallel for
      for (size_t i = 0; i < operator2.size(); i++) {
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator2[i], num_active_orbs, parity_check,tmp_stateV[i]);

      std::cout << "tmp_state 2-body[" << i << "] = \n" << tmp_stateV[i] << std::endl;
      }
      #pragma omp parallel for
      for (size_t i = 0; i < operator4.size(); i++) {
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator4[i], num_active_orbs, parity_check,tmp_stateV[i + operator2.size()]);
      }

      #pragma omp parallel for
      for (size_t i = 0; i < operator6.size(); i++) {
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator6[i], num_active_orbs, parity_check,tmp_stateV[i + operator2.size() + operator4.size()]);
      }
      #pragma omp parallel for
      for (size_t i = 0; i < operator8.size(); i++) {
      apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                              operator8[i], num_active_orbs, parity_check,tmp_stateV[i + operator2.size() + operator4.size() + operator6.size()]);
      }

      for (size_t i = 0; i < tmp_stateV.size(); i++) {

      tmp_state += tmp_stateV[i];
      }
    }


  return tmp_state;
}


PYBIND11_MODULE(fermionic_ops_cuda, m) {
  m.doc() = "Fermionic operator loop (opLoop) exposed via pybind11";

  m.def("op_loop", &py_opLoop, py::arg("op_folded_operators"),
        py::arg("num_active_orbs"), py::arg("parity_check"), py::arg("idx2det"),
        py::arg("det2idx"), py::arg("do_unsafe"), py::arg("state"),
        R"doc(
        Apply a sum of fermionic operator strings to a CI state vector.

        Parameters
        ----------
        op_folded_operators : dict
            Maps each operator string to its scalar prefactor.
            Key  : tuple of (orb_idx: int, is_creation: bool) tuples.
            Value: float prefactor.
        num_active_orbs : int
            Number of active spatial orbitals.
        parity_check : np.ndarray[uint64]
            Parity mask for each orbital index.
        idx2det : np.ndarray[uint64]
            Maps determinant index → bitstring.
        det2idx : np.ndarray[int32]
            Direct lookup table: bitstring → determinant index (-1 if absent).
        state : np.ndarray[float64]
            Input CI coefficient vector.

        Returns
        -------
        np.ndarray[float64]
            Output CI coefficient vector after applying all operators.
        )doc",
        py::return_value_policy::move);
}
