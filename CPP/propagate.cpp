#include <cassert>
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
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "fermionic_operator.hpp"
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
/* -------- Bitcount (GCC/Clang) -------- */
static inline int bitcount(uint64_t x) { return __builtin_popcountll(x); }

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

Eigen::MatrixXd apply_operator_SA_c(const Eigen::MatrixXd &state,
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

typedef struct {
  det2idx idx2det num_active_elec_alpha num_active_elec_beta num_active_orbs
      num_inactive_orbs num_virtual_orbs space_extension_offset
} CI_Info;

Eigen::MatrixXd
py_propagate_state_SA(const py::dict py_ops, const int num_active_orbs,
                      const py::array_t<uint64_t> py_parity_check,
                      const py::array_t<uint64_t> py_idx2det,
                      const py::dict py_det2idx, const bool do_unsafe,
                      const py::EigenDRef<Eigen::MatrixXd> py_state) {}

// py::array_t<double> py_propagate_state(const py::list& operators,
//                                       const py::array_t<double>& py_state,
//                                       const py::object& ci_info,
//                                       const py::object& thetas,
//                                       const py::object& wf_struct,
//                                       bool do_folding = true,
//                                       bool do_unsafe = false) {
//     // Extract info from ci_info
//     int num_inactive_orbs = ci_info.attr("num_inactive_orbs").cast<int>();
//     int num_active_orbs = ci_info.attr("num_active_orbs").cast<int>();
//     int num_virtual_orbs = ci_info.attr("num_virtual_orbs").cast<int>();
//     py::array_t<uint64_t> py_idx2det =
//     ci_info.attr("idx2det").cast<py::array_t<uint64_t>>(); py::dict
//     py_det2idx = ci_info.attr("det2idx").cast<py::dict>();

//     if (operators.empty()) {
//         return py_state;
//     }

//     // Initialize new_state as a copy of py_state
//     // We use Eigen for internal matrix operations if needed, but here we
//     work with numpy arrays mostly py::array_t<double> new_state =
//     py_state.attr("copy")();

//     // Create parity_check bitstrings
//     std::vector<uint64_t> parity_check(2 * num_active_orbs + 1, 0);
//     uint64_t num = 0;
//     for (int i = 0; i < 2 * num_active_orbs; ++i) {
//         num += (1ULL << i);
//         parity_check[i+1] = num;
//     }
//     // Reverse parity check order to match Python: parity_check[2 *
//     num_active_orbs - i] = num std::vector<uint64_t> parity_check_mapped(2 *
//     num_active_orbs + 1, 0); num = 0; for (int i = 2 * num_active_orbs - 1; i
//     >= 0; --i) {
//         num += (1ULL << i);
//         parity_check_mapped[2 * num_active_orbs - i] = num;
//     }

//     // Import construct_ups_state_SA from Python if needed
//     py::module_ osa =
//     py::module_::import("slowquant.unitary_coupled_cluster.operator_state_algebra");
//     py::function construct_ups_state_SA = osa.attr("construct_ups_state_SA");

//     // Loop over operators in reverse order
//     for (int i = static_cast<int>(operators.size()) - 1; i >= 0; --i) {
//         py::object op = operators[i];

//         if (py::isinstance<py::str>(op)) {
//             std::string op_str = op.cast<std::string>();
//             if (op_str != "U" && op_str != "Ud") {
//                 throw py::value_error(fmt::format("Unknown str operator,
//                 expected ('U', 'Ud') got {}", op_str));
//             }
//             bool dagger = (op_str == "Ud");
//             // Call Python construct_ups_state_SA
//             new_state = construct_ups_state_SA(new_state, ci_info, thetas,
//             wf_struct, dagger);
//         } else {
//             // FermionicOperator
//             py::object op_folded;
//             if (do_folding) {
//                 op_folded = op.attr("get_folded_operator")(num_inactive_orbs,
//                 num_active_orbs, num_virtual_orbs);
//             } else {
//                 op_folded = op;
//             }

//             // Call py_opLoop (already implemented and exposed in this
//             module) py::dict folded_operators =
//             op_folded.attr("operators").cast<py::dict>();
//             // Note: py_opLoop expects EigenDRef<Eigen::MatrixXd>, so we need
//             to cast new_state
//             // or just call it as a bound function if possible.
//             // Since py_opLoop is in this file, we can call it.
//             // But py_opLoop takes pybind11 types.

//             // We need to convert new_state to Eigen::MatrixXd for py_opLoop
//             or just call it via pybind11 new_state =
//             py_opLoop(folded_operators, num_active_orbs,
//             py::cast(parity_check_mapped),
//                                  py_idx2det, py_det2idx, do_unsafe,
//                                  new_state.cast<py::EigenDRef<Eigen::MatrixXd>>());
//         }
//     }

//     return new_state;
// }

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
}

#endif // PYBIND11_BUILD
