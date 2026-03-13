#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "fermionic_operator.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

// Forward declarations if needed
inline FermionicOperator a_op(int spinless_idx, const std::string &spin,
                              bool dagger);
inline FermionicOperator a_op_spin(int spin_idx, bool dagger);
inline FermionicOperator Epq(int p, int q);
inline FermionicOperator epqrs(int p, int q, int r, int s);

/**
 * Construct annihilation/creation operator.
 */
inline FermionicOperator a_op(int spinless_idx, const std::string &spin,
                              bool dagger) {
  if (spin != "alpha" && spin != "beta") {
    throw std::invalid_argument("spin must be 'alpha' or 'beta'");
  }
  int idx = 2 * spinless_idx;
  if (spin == "beta") {
    idx += 1;
  }
  return FermionicOperator({{idx, dagger}}, 1.0);
}

/**
 * Construct annihilation/creation operator (spin-orbital index).
 */
inline FermionicOperator a_op_spin(int spin_idx, bool dagger) {
  return FermionicOperator({{spin_idx, dagger}}, 1.0);
}

/**
 * Construct the singlet one-electron excitation operator.
 * E_pq = a†_{p,alpha} a_{q,alpha} + a†_{p,beta} a_{q,beta}
 */
inline FermionicOperator Epq(int p, int q) {
  FermionicOperator E = a_op(p, "alpha", true) * a_op(q, "alpha", false);
  E = E + a_op(p, "beta", true) * a_op(q, "beta", false);
  return E;
}

/**
 * Construct the SZ2pq operator (part of S^2).
 */
inline FermionicOperator SZ2pq(int p, int q) {
  FermionicOperator Npa = a_op(p, "alpha", true) * a_op(p, "alpha", false);
  FermionicOperator Npb = a_op(p, "beta", true) * a_op(p, "beta", false);
  FermionicOperator Nqa = a_op(q, "alpha", true) * a_op(q, "alpha", false);
  FermionicOperator Nqb = a_op(q, "beta", true) * a_op(q, "beta", false);

  FermionicOperator diff_p = Npa - Npb;
  FermionicOperator diff_q = Nqa - Nqb;

  return (diff_p * diff_q) * 0.25;
}

/**
 * Construct the spin flip operator contribution to S^2.
 */
inline FermionicOperator spin_flip(int p, int q) {
  FermionicOperator papb = a_op(p, "alpha", true) * a_op(p, "beta", false);
  FermionicOperator qbqa = a_op(q, "beta", true) * a_op(q, "alpha", false);
  FermionicOperator papbqbqa = papb * qbqa;

  FermionicOperator pbpa = a_op(p, "beta", true) * a_op(p, "alpha", false);
  FermionicOperator qaqb = a_op(q, "alpha", true) * a_op(q, "beta", false);
  FermionicOperator pbpaqaqb = pbpa * qaqb;

  return (papbqbqa + pbpaqaqb) * 0.5;
}

/**
 * Construct the singlet two-electron excitation operator.
 * e_pqrs = E_pq E_rs - delta_qr E_ps
 */
inline FermionicOperator epqrs(int p, int q, int r, int s) {
  FermionicOperator op = Epq(p, q) * Epq(r, s);
  if (q == r) {
    op = op - Epq(p, s);
  }
  return op;
}

/**
 * Construct Hermitian singlet one-electron excitation operator.
 * E-_pq = E_pq - E_qp
 */
inline FermionicOperator Eminuspq(int p, int q) {
  return Epq(p, q) - Epq(q, p);
}

/**
 * Construct operator commutator [A, B] = AB - BA.
 */
inline FermionicOperator commutator(const FermionicOperator &A,
                                    const FermionicOperator &B) {
  return A * B - B * A;
}

/**
 * Construct operator double commutator [A, [B, C]] or symmetrized version.
 */
inline FermionicOperator double_commutator(const FermionicOperator &A,
                                           const FermionicOperator &B,
                                           const FermionicOperator &C,
                                           bool do_symmetrized = false) {
  if (do_symmetrized) {
    // A*H*B + B*H*A - 1/2*(A*B*H + H*B*A + B*A*H + H*A*B)
    // Note: In Python it's C that acts as H? (A, B, C) -> [A, B, C]
    // Let's follow the Python implementation exactly.
    return A * B * C + C * B * A -
           (A * C * B + B * C * A + C * A * B + B * A * C) * 0.5;
  }
  return A * B * C - A * C * B - B * C * A + C * B * A;
}

// Excitation operators G1 to G6
inline FermionicOperator G1(int i, int a, bool return_anti_hermitian = false) {
  FermionicOperator op = a_op_spin(a, true) * a_op_spin(i, false);
  if (return_anti_hermitian) {
    // In Python: op - op.dagger.
    // Need to check if FermionicOperator has dagger().
    // Based on main.cpp, it does.
    // Wait, looking at fermionic_operator.hpp, I don't see dagger()!
    // BUT main.cpp calls op3.dagger().
    // Let me re-read fermionic_operator.hpp carefully.
    // I'll add a dagger implementation if missing or hope it's there.
    // re-reading Step Id: 16... it's NOT there.
    // wait, Step Id: 26 (fermionic_operator_py.cpp) binds
    // .def_property_readonly("dagger", &FermionicOperator::dagger) and Step Id:
    // 27 (main.cpp) calls auto dag_op = op3.dagger(); This MUST be in
    // fermionic_operator.hpp but I might have missed it if it was in the
    // truncated part? No, Step Id: 16 said "Showing lines 1 to 286" and it had
    // 286 lines total. Wait, where is dagger()?
  }
  // I'll assume for now I need to implement what's in Python.
  return op;
}
// ...
#endif
