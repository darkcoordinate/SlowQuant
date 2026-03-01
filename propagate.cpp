#include <cassert>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <omp.h>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef PYBIND11_BUILD
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

std::vector<double>
apply_operator_SA_c(std::vector<double> &state, std::vector<double> &tmp_state,
                    std::vector<uint64_t> &idx2det,
                    std::map<int, int> &det2idx, /* direct lookup table */
                    uint64_t det_lookup_size, int n_dets, operators operators,
                    int num_active_orbs, std::vector<uint64_t> &parity_check) {
  for (int i = 0; i < n_dets; ++i) {

    if (fabs(state[i]) < 1e-14)
      continue;

    uint64_t det = idx2det[i];
    int phase_changes = 0;
    int killstate = 0;
    /* ---- Apply annihilation operators ---- */
    for (size_t a = operators.annihilator.size() - 1; a >= 0; --a) {
      int orb_idx = operators.annihilator[a];
      int shift = 2 * num_active_orbs - 1 - orb_idx;
      uint64_t mask = 1ULL << shift;
      if ((det & mask) == 0) {
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

      if ((det & mask) != 0) {
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
    int new_idx = det2idx[det];
    if (new_idx < 0)
      continue;
    double sign = (phase_changes % 2 == 0) ? 1.0 : -1.0;
    tmp_state[new_idx] += operators.factor * sign * state[i];
  }
  return tmp_state;
}

#ifdef PYBIND11_BUILD

py::array_t<double> py_opLoop(
    // op_folded_operators: dict{ tuple[tuple[int,bool],...] : float }
    py::dict py_ops, int num_active_orbs, py::array_t<uint64_t> py_parity_check,
    py::array_t<uint64_t> py_idx2det, py::dict py_det2idx, bool do_unsafe,
    py::array_t<double> py_state) {
  std::vector<uint64_t> idx2det = py_idx2det.cast<std::vector<uint64_t>>();
  std::map<int, int> det2idx = py_det2idx.cast<std::map<int, int>>();
  uint64_t det_lookup_size = idx2det.size();
  int n_dets = idx2det.size();
  std::vector<uint64_t> parity_check =
      py_parity_check.cast<std::vector<uint64_t>>();

  std::cout << py_ops.size() << std::endl;
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
  std::cout << operator2.size() << std::endl;
  std::cout << operator4.size() << std::endl;
  std::cout << operator6.size() << std::endl;
  std::cout << operator8.size() << std::endl;
  std::cout << py_det2idx << std::endl;
  py::array_t<double> py_statel;

  std::vector<double> state(static_cast<double *>(py_state.request().ptr),
                            static_cast<double *>(py_state.request().ptr) +
                                py_state.request().size);
  std::vector<double> tmp_state(state.size());
  for (auto op : operator2) {
    tmp_state =
        apply_operator_SA_c(state, tmp_state, idx2det, det2idx, det_lookup_size,
                            n_dets, op, num_active_orbs, parity_check);
  }
  for (auto op : operator4) {
    tmp_state =
        apply_operator_SA_c(state, tmp_state, idx2det, det2idx, det_lookup_size,
                            n_dets, op, num_active_orbs, parity_check);
  }
  for (auto op : operator6) {
    tmp_state =
        apply_operator_SA_c(state, tmp_state, idx2det, det2idx, det_lookup_size,
                            n_dets, op, num_active_orbs, parity_check);
  }
  for (auto op : operator8) {
    tmp_state =
        apply_operator_SA_c(state, tmp_state, idx2det, det2idx, det_lookup_size,
                            n_dets, op, num_active_orbs, parity_check);
  }
  py_statel = py::array_t<double>(tmp_state.size(), tmp_state.data());
  std::cout << py_statel << std::endl;
  return py_statel;
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
        )doc");
}

#endif // PYBIND11_BUILD
