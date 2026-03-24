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

#include "fermionic_operator.hpp"
//#include "operators.hpp"
int t1(const py::dict py_ops) {

  auto i = py_ops.cast<std::map<std::vector<std::tuple<int, bool>>, double>>();
  std::cout << i.size() << " : ";
  for (auto item : i) {
    // std::cout << item.first << " \n";
    for (auto ai : item.first) {
      // std::cout<<ai<<" \n";
      // auto pyl = ai.cast<std::tuple<int , bool >>();
      std::cout << std::get<0>(ai) << " " << std::get<1>(ai) << " ";
    }
    std::cout << item.second << " : ";
    // std::cout << item.second << " \n";
  }
  std::cout << std::endl;
  return 0;
}

__device__ __host__ static inline int bitcount(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}

/*
Apply fermionic operator to state (single-state version)

Parameters
----------
state           : input coefficient vector
tmp_state       : output vector (must be zeroed before call)
idx2det         : array mapping index -> determinant bitstring
det2idx         : lookup table mapping determinant -> index
det_lookup_size : size of det2idx lookup table
n_dets          : number of determinants
anni_idxs       : annihilation indices
n_anni          : number of annihilation operators
create_idxs     : creation indices
n_create        : number of creation operators
num_active_orbs : number of active spatial orbitals
parity_check    : parity masks
factor          : prefactor of operator
*/

typedef struct {
  std::vector<int> creator;
  std::vector<int> annihilator;
  double factor;
  int len;

  void print() const {
    std::cout << "Creator ";
    for (size_t i = 0; i < creator.size(); i++) {
      std::cout << creator[i] << " ";
    }
    std::cout << "Annhilator ";
    for (size_t i = 0; i < annihilator.size(); i++) {
      std::cout << annihilator[i] << " ";
    }
    std::cout << factor << " \n";
  }
} operators;

__device__ __host__ Eigen::MatrixXd apply_operator_SA_c(const Eigen::MatrixXd &state,
                                    const std::vector<uint64_t> &idx2det,
                                    const std::map<uint64_t, uint64_t> &det2idx,
                                    const uint64_t det_lookup_size,
                                    const int n_dets, const operators &ops,
                                    const int num_active_orbs,
                                    const std::vector<uint64_t> &parity_check) {

  Eigen::MatrixXd tmp_state2 =
      Eigen::MatrixXd::Zero(state.rows(), state.cols());
  for (int i = 0; i < n_dets; ++i) {
    bool is_non_zero = (state.col(i).array().abs() > 1e-14).any();
    if (!is_non_zero)
      continue;
    uint64_t det = idx2det[i];
    int phase_changes = 0;
    int killstate = 0;

    /* ---- Apply annihilation operators ---- */
    for (int a = static_cast<int>(ops.annihilator.size()) - 1; a >= 0; --a) {
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
    for (int a = static_cast<int>(ops.creator.size()) - 1; a >= 0; --a) {
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
  return tmp_state2;
}

#ifdef PYBIND11_BUILD

Eigen::MatrixXd py_opLoop(const py::dict py_ops, const int num_active_orbs,
                          const py::array_t<uint64_t> py_parity_check,
                          const py::array_t<uint64_t> py_idx2det,
                          const py::dict py_det2idx, const bool do_unsafe,
                          const py::EigenDRef<Eigen::MatrixXd> py_state) {
  std::vector<uint64_t> idx2det = py_idx2det.cast<std::vector<uint64_t>>();
  std::map<uint64_t, uint64_t> det2idx =
      py_det2idx.cast<std::map<uint64_t, uint64_t>>();
  uint64_t det_lookup_size = idx2det.size();
  int n_dets = idx2det.size();
  std::vector<uint64_t> parity_check =
      py_parity_check.cast<std::vector<uint64_t>>();

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
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 2;
      operator2.push_back(op);
    } else if (py_label.size() == 4) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 4;
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
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 6;

      operator6.push_back(op);
    } else if (py_label.size() == 8) {
      operators op;
      op.factor = item.second.cast<double>();
      for (py::size_t i = 0; i < py_label.size(); i++) {
        py::tuple py_op = py_label[i].cast<py::tuple>();
        int orb = py_op[0].cast<int>();
        bool is_creation = py_op[1].cast<bool>();
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 8;
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

#pragma omp parallel for
  for (size_t i = 0; i < operator2.size(); i++) {
    tmp_stateV[i] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator2[i], num_active_orbs, parity_check);
  }
#pragma omp parallel for
  for (size_t i = 0; i < operator4.size(); i++) {
    tmp_stateV[i + operator2.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator4[i], num_active_orbs, parity_check);
  }

#pragma omp parallel for
  for (size_t i = 0; i < operator6.size(); i++) {
    tmp_stateV[i + operator2.size() + operator4.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator6[i], num_active_orbs, parity_check);
  }
#pragma omp parallel for
  for (size_t i = 0; i < operator8.size(); i++) {
    tmp_stateV[i + operator2.size() + operator4.size() + operator6.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator8[i], num_active_orbs, parity_check);
  }

  for (size_t i = 0; i < tmp_stateV.size(); i++) {

    tmp_state += tmp_stateV[i];
  }

  return tmp_state;
}

__device__ __host__ Eigen::MatrixXd opLoop(const FermionicOperator &ops, const int num_active_orbs,
                       const std::vector<uint64_t> &parity_check,
                       const std::vector<uint64_t> &idx2det,
                       const std::map<uint64_t, uint64_t> &det2idx,
                       const bool do_unsafe, const Eigen::MatrixXd &state) {
  uint64_t det_lookup_size = idx2det.size();
  int n_dets = idx2det.size();

  std::vector<operators> operator2;
  std::vector<operators> operator4;
  std::vector<operators> operator6;
  std::vector<operators> operator8;
  for (auto item : ops.operators) {
    auto label = item.first;
    if (label.size() == 2) {
      operators op;
      op.factor = item.second;
      for (py::size_t i = 0; i < label.size(); i++) {
        auto py_op = label[i];
        int orb = std::get<0>(py_op);
        bool is_creation = std::get<1>(py_op);
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 2;
      operator2.push_back(op);
    } else if (label.size() == 4) {
      operators op;
      op.factor = item.second;
      for (py::size_t i = 0; i < label.size(); i++) {
        auto py_op = label[i];
        int orb = std::get<0>(py_op);
        bool is_creation = std::get<1>(py_op);
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 4;
      operator4.push_back(op);
    }

    else if (label.size() == 6) {
      operators op;
      op.factor = item.second;
      for (py::size_t i = 0; i < label.size(); i++) {
        auto py_op = label[i];
        int orb = std::get<0>(py_op);
        bool is_creation = std::get<1>(py_op);
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 6;

      operator6.push_back(op);
    } else if (label.size() == 8) {
      operators op;
      op.factor = item.second;
      for (py::size_t i = 0; i < label.size(); i++) {
        auto py_op = label[i];
        int orb = std::get<0>(py_op);
        bool is_creation = std::get<1>(py_op);
        if (is_creation)
          op.creator.push_back(orb);
        else
          op.annihilator.push_back(orb);
      }
      op.len = 8;
      operator8.push_back(op);
    } else {
      operators op;
      op.factor = item.second;
      op.len = 0;
      operator2.push_back(op);
    }
  }
  Eigen::MatrixXd tmp_state = Eigen::MatrixXd::Zero(state.rows(), state.cols());
  std::vector<Eigen::MatrixXd> tmp_stateV(operator2.size() + operator4.size() +
                                          operator6.size() + operator8.size());
  // std::cout << state.format(OctaveFmt) << std::endl;

  //#pragma omp parallel for
  for (size_t i = 0; i < operator2.size(); i++) {
    tmp_stateV[i] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator2[i], num_active_orbs, parity_check);
  }
  //#pragma omp parallel for
  for (size_t i = 0; i < operator4.size(); i++) {
    tmp_stateV[i + operator2.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator4[i], num_active_orbs, parity_check);
  }

  //#pragma omp parallel for
  for (size_t i = 0; i < operator6.size(); i++) {
    tmp_stateV[i + operator2.size() + operator4.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator6[i], num_active_orbs, parity_check);
  }
  //#pragma omp parallel for
  for (size_t i = 0; i < operator8.size(); i++) {
    tmp_stateV[i + operator2.size() + operator4.size() + operator6.size()] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator8[i], num_active_orbs, parity_check);
  }

  for (size_t i = 0; i < tmp_stateV.size(); i++) {

    tmp_state += tmp_stateV[i];
  }

  return tmp_state;
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
    det2idx = py_ci_info.attr("det2idx").cast<std::map<uint64_t, uint64_t>>();
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

Eigen::MatrixXd py_propagate_state_SA(
    py::list py_ops, const py::EigenDRef<Eigen::MatrixXd> state,
    const py::object &py_ci_info, const py::array_t<double> py_thetas,
    const py::object &py_wf_struct, py::bool_ py_do_folding) {
  CI_Info ci_info(py_ci_info);
  Eigen::MatrixXd state_ = state;
  std::vector<uint64_t> parity_check(2 * ci_info.num_active_orbs + 1);
  uint64_t num = 0;
  for (int i = 2 * ci_info.num_active_orbs - 1; i >= 0; i--) {
    num += 1 << i;
    parity_check[2 * ci_info.num_active_orbs - i] = num;
  }

  for (int i = py_ops.size(); i > 0; i--) {
    // Eigen::MatrixXd tmp_state =
    //     Eigen::MatrixXd::Zero(state.rows(), state.cols());

    auto a = py_ops[i - 1]
                 .attr("operators")
                 .cast<std::map<std::vector<std::tuple<int, bool>>, double>>();
    FermionicOperator c1(a);

    if (py_do_folding) {
      c1 = c1.get_folded_operator(ci_info.num_inactive_orbs,
                                  ci_info.num_active_orbs,
                                  ci_info.num_virtual_orbs);
    }

    auto tmp_state = opLoop(c1, ci_info.num_active_orbs, parity_check,
                            ci_info.idx2det, ci_info.det2idx, false, state_);
    state_ = tmp_state;
  }

  return state_;
}

__device__ __host__ Eigen::MatrixXd propagate_state_SA(std::vector<FermionicOperator> py_ops,
                                   const Eigen::MatrixXd state,
                                   const CI_Info &py_ci_info,
                                   const std::vector<double> py_thetas,
                                   const py::object &py_wf_struct,
                                   bool py_do_folding) {
  CI_Info ci_info(py_ci_info);
  Eigen::MatrixXd state_ = state;
  std::vector<uint64_t> parity_check(2 * ci_info.num_active_orbs + 1);
  uint64_t num = 0;
  for (int i = 2 * ci_info.num_active_orbs - 1; i >= 0; i--) {
    num += 1 << i;
    parity_check[2 * ci_info.num_active_orbs - i] = num;
  }

  for (int i = py_ops.size(); i > 0; i--) {
    // Eigen::MatrixXd tmp_state =
    //     Eigen::MatrixXd::Zero(state.rows(), state.cols());

    auto a = py_ops[i - 1];
    FermionicOperator c1(a);

    if (py_do_folding) {
      c1 = c1.get_folded_operator(ci_info.num_inactive_orbs,
                                  ci_info.num_active_orbs,
                                  ci_info.num_virtual_orbs);
    }

    auto tmp_state = opLoop(c1, ci_info.num_active_orbs, parity_check,
                            ci_info.idx2det, ci_info.det2idx, false, state_);
    state_ = tmp_state;
  }

  return state_;
}

Eigen::MatrixXd expectation_vector_SA_py(py::EigenDRef<Eigen::MatrixXd> bra,
                                         const py::list &py_ops,
                                         py::EigenDRef<Eigen::MatrixXd> ket,
                                         const CI_Info &ci_info,
                                         const py::array_t<double> &py_thetas,
                                         const py::object &py_wf_struct,
                                         py::bool_ py_do_folding) {
  std::vector<FermionicOperator> ops;
  for (size_t i = 0; i < py_ops.size(); i++) {
    ops.push_back(FermionicOperator(
        py_ops[i]
            .attr("operators")
            .cast<std::map<std::vector<std::tuple<int, bool>>, double>>()));
  }
  auto op_ket = propagate_state_SA(ops, ket, ci_info,
                                   py_thetas.cast<std::vector<double>>(),
                                   py_wf_struct, py_do_folding);
  return bra * op_ket.transpose();
}

__device__ __host__ Eigen::MatrixXd expectation_vector_SA(const Eigen::MatrixXd bra,
                                      const std::vector<FermionicOperator> ops,
                                      const Eigen::MatrixXd ket,
                                      const CI_Info &ci_info,
                                      const std::vector<double> &thetas,
                                      const py::object &py_wf_struct,
                                      bool py_do_folding) {
  auto op_ket = propagate_state_SA(ops, ket, ci_info, thetas, py_wf_struct,
                                   py_do_folding);
  return bra * op_ket.transpose();
}

__global__ void derivative_theta_kernel(const Eigen::MatrixXd *bra,
                                      const std::vector<FermionicOperator> *ops,
                                      const Eigen::MatrixXd *ket,
                                      const CI_Info &ci_info,
                                      const std::vector<double> &thetas,
                                      const py::object &py_wf_struct,
                                      bool py_do_folding) {
  
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

Eigen::MatrixXd test_SA(py::list py_ops,
                        const py::EigenDRef<Eigen::MatrixXd> state,
                        const py::object &py_ci_info, const py::list py_thetas,
                        const py::object py_wf_struc, py::bool_ py_do_folding) {
  std::cout << "************************" << std::endl;
  for (size_t i = 0; i < py_ops.size(); i++) {
    py::object op = py_ops[i];
    std::cout << op << std::endl;
  }
  CI_Info ci_info(py_ci_info);
  std::cout << state << std::endl;
  std::cout << ci_info.num_active_orbs << std::endl;
  std::cout << ci_info.num_active_elec_alpha << std::endl;
  std::cout << ci_info.num_active_elec_beta << std::endl;
  std::cout << ci_info.num_inactive_orbs << std::endl;
  std::cout << ci_info.num_virtual_orbs << std::endl;
  std::cout << ci_info.space_extension_offset << std::endl;
  std::cout << "************************" << std::endl;
  auto thetas = py_thetas.cast<std::vector<double>>();
  for (size_t i = 0; i < thetas.size(); i++) {
    std::cout << "theta: " << i << " = " << thetas[i] << std::endl;
  }
  Eigen::MatrixXd tmp_state = Eigen::MatrixXd::Zero(state.rows(), state.cols());
  return tmp_state;
}

PYBIND11_MODULE(fermionic_ops, m) {
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
  m.def("t1", &t1);
  m.def("propagate_state_SA_cpp", &py_propagate_state_SA, py::arg("operators"),
        py::arg("ci_coeffs"), py::arg("ci_info"), py::arg("thetas"),
        py::arg("wf_struct"), py::arg("do_folding") = true,
        R"doc(
        Apply a sum of fermionic operator strings to a CI state vector.

        Parameters
        ----------
        operators : list
            List of fermionic operators to apply.
        ci_info : dict
            Information about the CI space.
        thetas : dict
            Thetas for the UCC operators.
        wf_struct : dict
            Structure of the wave function.
        do_folding : bool
            Whether to fold the operators.
        do_unsafe : bool
            Whether to use unsafe mode.

        Returns
        -------
        np.ndarray[float64]
            Output CI coefficient vector after applying all operators.
        )doc");
  m.def("test_SA", &test_SA, py::arg("operators"), py::arg("ci_coeffs"),
        py::arg("ci_info"), py::arg("thetas"), py::arg("wf_struct"),
        py::arg("do_folding"), "good", py::return_value_policy::move);
  m.def("derivative_theta_ket", &derivative_theta_ket, py::arg("bra"),
        py::arg("op1"), py::arg("op2"), py::arg("ket"), py::arg("ci_info"),
        py::arg("thetas"), py::arg("wf_struct"), py::arg("do_folding") = true,
        py::arg("specific_state"), "good", py::return_value_policy::move);
}

#endif // PYBIND11_BUILD
