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
} operators;

Eigen::MatrixXd apply_operator_SA_c(
    const Eigen::MatrixXd &state, const std::vector<uint64_t> &idx2det,
    const std::map<int, int> &det2idx, /* direct lookup table */
    const uint64_t det_lookup_size, const int n_dets, const operators operators,
    const int num_active_orbs, const std::vector<uint64_t> &parity_check) {
  Eigen::MatrixXd tmp_state2 =
      Eigen::MatrixXd::Zero(state.rows(), state.cols());
  for (int i = 0; i < n_dets; ++i) {
    bool is_non_zero = false;
    for (auto a : state.col(i)) {
      if (std::abs(a) > 1e-14) {
        is_non_zero = true;
        break;
      }
    }
    if (!is_non_zero)
      continue;
    // if (state.col(i).cwiseAbs().maxCoeff() < 1e-14)
    //   break;

    uint64_t det = idx2det[i];
    int phase_changes = 0;
    int killstate = 0;
    /* ---- Apply annihilation operators ---- */
    for (size_t a = operators.annihilator.size() - 1; a >= 0; --a) {
      int orb_idx = operators.annihilator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if (((det >> shift) & 1) == 0) {
        killstate = 1;
        break;
      }
      std::cout << "determinant: " << fmt::format("0x{:016b}", det)
                << std::endl;
      std::cout << "mask       : " << fmt::format("0x{:016b}", mask)
                << std::endl;
      det ^= mask;
      std::cout << "determinant: " << fmt::format("0x{:016b}", det)
                << std::endl;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;
    /* ---- Apply creation operators ---- */
    for (int a = operators.creator.size() - 1; a >= 0; --a) {
      int orb_idx = operators.creator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;

      if (((det >> shift) & 1) == 1) {
        killstate = 1;
        break;
      }
      std::cout << "determinant: " << fmt::format("0x{:016b}", det)
                << std::endl;
      std::cout << "mask       : " << fmt::format("0x{:016b}", mask)
                << std::endl;
      det ^= mask;
      std::cout << "determinant: " << fmt::format("0x{:016b}", det)
                << std::endl;
      phase_changes += bitcount(det & parity_check[orb_idx]);
    }
    if (killstate)
      continue;
    /* ---- Lookup new determinant index ---- */
    if (det >= det_lookup_size)
      continue;
    int new_idx = det2idx.at(det);
    if (new_idx < 0)
      continue;
    double sign = (phase_changes % 2 == 0) ? 1.0 : -1.0;
    std::cout << "new_idx: " << new_idx << std::endl;
    std::cout << "sign: " << sign << std::endl;
    std::cout << "operators.factor: " << operators.factor << std::endl;
    std::cout << "state.col(i): " << state.col(i) << std::endl;
    std::cout << "operators.factor * sign * state.col(i): "
              << operators.factor * sign * state.col(i) << std::endl;
    tmp_state2.col(new_idx) += operators.factor * sign * state.col(i);
  }
  return tmp_state2;
}

#ifdef PYBIND11_BUILD

Eigen::MatrixXd py_opLoop(
    // op_folded_operators: dict{ tuple[tuple[int,bool],...] : float }
    const py::dict py_ops, const int num_active_orbs,
    const py::array_t<uint64_t> py_parity_check,
    const py::array_t<uint64_t> py_idx2det, const py::dict py_det2idx,
    const bool do_unsafe, const py::EigenDRef<Eigen::MatrixXd> py_state) {

  std::vector<uint64_t> idx2det = py_idx2det.cast<std::vector<uint64_t>>();
  std::map<int, int> det2idx = py_det2idx.cast<std::map<int, int>>();
  uint64_t det_lookup_size = idx2det.size();
  int n_dets = idx2det.size();
  std::vector<uint64_t> parity_check =
      py_parity_check.cast<std::vector<uint64_t>>();

  // std::cout << py_ops.size() << std::endl;
  // collect all operators in 4 different types of vector depenping  on
  // py_label_list size
  // first one is creation and annihilation
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
    }

    if (py_label.size() == 4) {
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
      operator4.push_back(op);
    }

    if (py_label.size() == 6) {
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
      operator6.push_back(op);
    }
    if (py_label.size() == 8) {
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
      operator8.push_back(op);
    }
  }
  // std::cout << operator2.size() << std::endl;
  // std::cout << operator4.size() << std::endl;
  // std::cout << operator6.size() << std::endl;
  // std::cout << operator8.size() << std::endl;
  // std::cout << py_det2idx << std::endl;
  Eigen::MatrixXd state = py_state;
  Eigen::MatrixXd tmp_state(state.rows(), state.cols());
  std::vector<Eigen::MatrixXd> tmp_stateV(operator2.size() + operator4.size() +
                                          operator6.size() + operator8.size());
  //#pragma omp parallel for
  for (size_t i = 0; i < operator2.size(); i++) {
    tmp_stateV[i] =
        apply_operator_SA_c(state, idx2det, det2idx, det_lookup_size, n_dets,
                            operator2[i], num_active_orbs, parity_check);
  }

  // std::cout << "tmp_stateV size: " << tmp_stateV.size() << std::endl;

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
  //#pragma omp parallel for
  for (size_t i = 0; i < tmp_stateV.size(); i++) {
    tmp_state += tmp_stateV[i];
  }

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
}

#endif // PYBIND11_BUILD
