#ifndef FERMIONIC_OPERATOR_H
#define FERMIONIC_OPERATOR_H
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <omp.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
class FermionicOperator;
FermionicOperator do_extended_normal_ordering(const FermionicOperator &fermi);

using OperatorString = std::vector<std::tuple<int, bool>>;

struct OperatorStringCompare {
  bool operator()(const OperatorString &lhs, const OperatorString &rhs) const {
    return lhs < rhs;
  }
};

class FermionicOperator {
public:
  std::map<std::vector<std::tuple<int, bool>>, double> operators;
  FermionicOperator() = default;
  explicit FermionicOperator(const std::vector<std::tuple<int, bool>> &ops,
                             const double fac) {
    operators.insert({ops, fac});
  }

  explicit FermionicOperator(
      const std::map<std::vector<std::tuple<int, bool>>, double> &ops) {
    operators = ops;
  }

  FermionicOperator operator+(const FermionicOperator &other) const {

    FermionicOperator result;
    result.operators = operators;
    for (const auto &[ops, fac] : other.operators) {
      auto it = result.operators.find(ops);
      if (it != result.operators.end()) {
        result.operators[ops] += fac;
        if (std::abs(result.operators[ops]) < 1e-14)
          result.operators.erase(ops);
      } else {
        result.operators[ops] = fac;
      }
    }
    return result;
  }

  FermionicOperator operator-(const FermionicOperator &other) const {
    FermionicOperator result;
    result.operators = operators;
    for (const auto &[ops, fac] : other.operators) {
      result.operators[ops] -= fac;
      if (std::abs(result.operators[ops]) < 1e-14)
        result.operators.erase(ops);
    }
    return result;
  }

  FermionicOperator &operator-=(const FermionicOperator &other) {
    for (const auto &[ops, fac] : other.operators) {
      operators[ops] -= fac;
      if (std::abs(operators[ops]) < 1e-14)
        operators.erase(ops);
    }
    return *this;
  }

  FermionicOperator operator*(double factor) const {
    FermionicOperator result;
    if (std::abs(factor) < 1e-14)
      return result;
    for (const auto &[ops, fac] : operators) {
      double val = fac * factor;
      if (std::abs(val) > 1e-14) {
        result.operators[ops] = val;
      }
    }
    return result;
  }

  // Multiply two FermionicOperator objects by performing normal ordering on the
  // product of their operator strings.
  FermionicOperator operator*(const FermionicOperator &other) const {
    FermionicOperator result;
    // Iterate over each operator string in the left operand.
    for (const auto &[ops_left, fac_left] : this->operators) {
      // Iterate over each operator string in the right operand.
      for (const auto &[ops_right, fac_right] : other.operators) {
        // Concatenate the operator vectors (left followed by right).
        std::vector<std::tuple<int, bool>> combined = ops_left;
        combined.insert(combined.end(), ops_right.begin(), ops_right.end());
        // Create a temporary FermionicOperator for this combined string with
        // the product of coefficients.
        FermionicOperator temp({combined}, fac_left * fac_right);
        // Perform extended normal ordering on the combined operator.
        FermionicOperator ordered = do_extended_normal_ordering(temp);
        // Merge the resulting ordered terms into the result.
        for (const auto &[new_ops, new_fac] : ordered.operators) {
          result.operators[new_ops] += new_fac;
          // Remove near-zero coefficients to keep the representation clean.
          if (std::abs(result.operators[new_ops]) < 1e-14) {
            result.operators.erase(new_ops);
          }
        }
      }
    }
    return result;
  }

  FermionicOperator get_folded_operator(const int num_inactive_orbs,
                                        const int num_active_orbs,
                                        const int num_virtual_orbs) {
    std::map<std::vector<std::tuple<int, bool>>, double> folded_operators;
    std::vector<int> inactive_idx;
    std::vector<int> active_idx;
    std::vector<int> virtual_idx;
    for (int i = 0;
         i < 2 * num_inactive_orbs + 2 * num_active_orbs + 2 * num_virtual_orbs;
         i++) {
      if (i < 2 * num_inactive_orbs) {
        inactive_idx.push_back(i);
      } else if (i < 2 * num_inactive_orbs + 2 * num_active_orbs) {
        active_idx.push_back(i);
      } else {
        virtual_idx.push_back(i);
      }
    }

    for (auto &[ops, fac_] : operators) {
      std::vector<int> virtual_;
      std::vector<int> virtual_dagger;
      std::vector<int> inactive;
      std::vector<int> inactive_dagger;
      std::vector<std::tuple<int, bool>> active;
      std::vector<std::tuple<int, bool>> active_dagger;
      double fac = 1;

      for (auto anni : ops) {
        int idx = std::get<0>(anni);
        bool dagger = std::get<1>(anni);
        if (dagger) {
          if (std::find(inactive_idx.begin(), inactive_idx.end(), idx) !=
              inactive_idx.end()) {
            inactive_dagger.push_back(idx);
          } else if (std::find(active_idx.begin(), active_idx.end(), idx) !=
                     active_idx.end()) {
            active_dagger.push_back({idx - 2 * num_inactive_orbs, dagger});
          } else {
            virtual_dagger.push_back(idx);
          }
        } else {
          if (std::find(inactive_idx.begin(), inactive_idx.end(), idx) !=
              inactive_idx.end()) {
            inactive.push_back(idx);
          } else if (std::find(active_idx.begin(), active_idx.end(), idx) !=
                     active_idx.end()) {
            active.push_back({idx - 2 * num_inactive_orbs, dagger});
          } else {
            virtual_.push_back(idx);
          }
        }
      }
      if (virtual_.size() != 0 || virtual_dagger.size() != 0) {
        continue;
      }
      // auto active_op = active_dagger + active;
      auto active_op = active_dagger;
      active_op.insert(active_op.end(), active.begin(), active.end());
      auto bra_side = inactive_dagger;
      auto ket_side = inactive;

      if (bra_side != ket_side) {
        continue;
      }
      if (inactive_dagger.size() % 2 == 1 && active_dagger.size() % 2 == 1) {
        fac *= -1;
      }
      double ket_flip_fac = 1;
      for (int i = 1; i < ket_side.size() + 1; i++) {
        if (i % 2 == 0) {
          ket_flip_fac *= -1;
        }
      }
      fac *= ket_flip_fac;
      const auto new_key = active_op;
      auto it = folded_operators.find(new_key);
      if (it != folded_operators.end()) {
        folded_operators[new_key] += fac * operators[ops];
      } else {
        folded_operators[new_key] = fac * operators[ops];
      }
    }
    return FermionicOperator(folded_operators);
  }
};

FermionicOperator do_extended_normal_ordering(const FermionicOperator &fermi) {
  // Queue of operator strings and their coefficients.
  std::vector<std::vector<std::tuple<int, bool>>> operator_queue;
  std::vector<double> factor_queue;
  // Initialize queues from the input operator map.
  for (const auto &[ops, fac] : fermi.operators) {
    operator_queue.push_back(ops);
    factor_queue.push_back(fac);
  }
  // Result map to accumulate ordered operators.
  std::map<std::vector<std::tuple<int, bool>>, double> new_operators;
  // Process until queue is empty.
  while (!operator_queue.empty()) {
    // Pop front.
    std::vector<std::tuple<int, bool>> next_operator = operator_queue.front();
    double factor = factor_queue.front();
    operator_queue.erase(operator_queue.begin());
    factor_queue.erase(factor_queue.begin());
    // Perform bubble‑style passes until no changes or term vanishes.
    while (true) {
      std::size_t current_idx = 0;
      bool changed = false;
      bool is_zero = false;
      while (true) {
        if (next_operator.empty() || current_idx + 1 >= next_operator.size())
          break;
        auto a = next_operator[current_idx];
        auto b = next_operator[current_idx + 1];
        int a_idx = std::get<0>(a);
        int b_idx = std::get<0>(b);
        bool a_cre = std::get<1>(a);
        bool b_cre = std::get<1>(b);
        // Both creation operators.
        if (a_cre && b_cre) {
          if (a_idx == b_idx) {
            is_zero = true;
          } else if (a_idx < b_idx) {
            std::swap(next_operator[current_idx],
                      next_operator[current_idx + 1]);
            factor = -factor;
            changed = true;
          }
        }
        // Annihilation then creation.
        else if (!a_cre && b_cre) {
          if (a_idx == b_idx) {
            // Cancel both operators.
            std::vector<std::tuple<int, bool>> reduced = next_operator;
            reduced.erase(reduced.begin() + current_idx,
                          reduced.begin() + current_idx + 2);
            if (!reduced.empty()) {
              operator_queue.push_back(reduced);
              factor_queue.push_back(factor);
            }
            std::swap(next_operator[current_idx],
                      next_operator[current_idx + 1]);
            factor = -factor;
            changed = true;
          } else {
            std::swap(next_operator[current_idx],
                      next_operator[current_idx + 1]);
            factor = -factor;
            changed = true;
          }
        }
        // Creation then annihilation – nothing to do.
        else if (a_cre && !b_cre) {
          // pass
        }
        // Both annihilation operators.
        else {
          if (a_idx == b_idx) {
            is_zero = true;
          } else if (a_idx < b_idx) {
            std::swap(next_operator[current_idx],
                      next_operator[current_idx + 1]);
            factor = -factor;
            changed = true;
          }
        }
        ++current_idx;
        if (current_idx + 1 >= next_operator.size() || is_zero)
          break;
      }
      if (!changed || is_zero) {
        if (!is_zero) {
          auto it = new_operators.find(next_operator);
          if (it == new_operators.end()) {
            new_operators.emplace(next_operator, factor);
          } else {
            it->second += factor;
            if (std::abs(it->second) < 1e-14) {
              new_operators.erase(it);
            }
          }
        }
        break;
      }
    }
  }
  // Build FermionicOperator from the map.
  FermionicOperator result;
  result.operators = std::move(new_operators);
  return result;
}

#endif