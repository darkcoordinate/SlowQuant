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
  std::map<OperatorString, double> operators;
  FermionicOperator() = default;
  explicit FermionicOperator(const OperatorString &ops, const double fac) {
    if (std::abs(fac) > 1e-14) {
      operators.insert({ops, fac});
    }
  }

  explicit FermionicOperator(const std::map<OperatorString, double> &ops) {
    for (const auto &item : ops) {
      if (std::abs(item.second) > 1e-14) {
        operators.insert(item);
      }
    }
  }

  const std::map<OperatorString, double> &get_operators() const {
    return operators;
  }

  FermionicOperator operator+(const FermionicOperator &other) const {
    FermionicOperator result;
    result.operators = operators;
    for (auto it = other.operators.begin(); it != other.operators.end(); ++it) {
      const auto &ops = it->first;
      const auto fac = it->second;
      result.operators[ops] += fac;
      if (std::abs(result.operators[ops]) < 1e-14)
        result.operators.erase(ops);
    }
    return result;
  }

  FermionicOperator &operator+=(const FermionicOperator &other) {
    for (const auto &[ops, fac] : other.operators) {
      operators[ops] += fac;
      if (std::abs(operators[ops]) < 1e-14)
        operators.erase(ops);
    }
    return *this;
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

  FermionicOperator &operator*=(double factor) {
    if (std::abs(factor) < 1e-14) {
      operators.clear();
      return *this;
    }
    auto it = operators.begin();
    while (it != operators.end()) {
      it->second *= factor;
      if (std::abs(it->second) < 1e-14) {
        it = operators.erase(it);
      } else {
        ++it;
      }
    }
    return *this;
  }

  FermionicOperator operator-() const { return (*this) * -1.0; }

  bool operator==(const FermionicOperator &other) const {
    return operators == other.operators;
  }

  bool operator!=(const FermionicOperator &other) const {
    return !(*this == other);
  }

  FermionicOperator operator*(const FermionicOperator &other) const {
    FermionicOperator result;
    for (const auto &[ops_left, fac_left] : this->operators) {
      for (const auto &[ops_right, fac_right] : other.operators) {
        OperatorString combined = ops_left;
        combined.insert(combined.end(), ops_right.begin(), ops_right.end());
        FermionicOperator temp(combined, fac_left * fac_right);
        FermionicOperator ordered = do_extended_normal_ordering(temp);
        for (const auto &[new_ops, new_fac] : ordered.operators) {
          result.operators[new_ops] += new_fac;
          if (std::abs(result.operators[new_ops]) < 1e-14) {
            result.operators.erase(new_ops);
          }
        }
      }
    }
    return result;
  }

  FermionicOperator &operator*=(const FermionicOperator &other) {
    *this = (*this) * other;
    return *this;
  }

  FermionicOperator dagger() const {
    FermionicOperator result;
    for (auto it = operators.begin(); it != operators.end(); ++it) {
      const auto &ops = it->first;
      const auto fac = it->second;
      OperatorString dag_ops = ops;
      std::reverse(dag_ops.begin(), dag_ops.end());
      for (auto &op : dag_ops) {
        std::get<1>(op) = !std::get<1>(op);
      }
      result.operators[dag_ops] = fac;
    }
    return result;
  }

  std::map<std::string, double> operators_readable() const {
    std::map<std::string, double> result;
    for (const auto &[ops, fac] : operators) {
      std::string s = "";
      for (size_t i = 0; i < ops.size(); ++i) {
        s += std::to_string(std::get<0>(ops[i]));
        if (std::get<1>(ops[i]))
          s += "^";
        if (i < ops.size() - 1)
          s += " ";
      }
      if (s.empty())
        s = "I";
      result[s] = fac;
    }
    return result;
  }

  std::map<int, int> operator_count() const {
    std::map<int, int> counts;
    for (const auto &[ops, fac] : operators) {
      counts[ops.size()]++;
    }
    return counts;
  }

  std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>,
             std::vector<double>>
  get_info() const {
    std::vector<std::vector<int>> annihilations;
    std::vector<std::vector<int>> creations;
    std::vector<double> coefficients;
    for (const auto &[ops, fac] : operators) {
      std::vector<int> anni, crea;
      for (const auto &op : ops) {
        if (std::get<1>(op))
          crea.push_back(std::get<0>(op));
        else
          anni.push_back(std::get<0>(op));
      }
      annihilations.push_back(anni);
      creations.push_back(crea);
      coefficients.push_back(fac);
    }
    return {annihilations, creations, coefficients};
  }

  std::map<std::string, double> get_qiskit_form(int num_orbs) const {
    // Placeholder for Jordan-Wigner or similar mapping.
    // For now, return a simplified representation as expected by main.cpp.
    return operators_readable();
  }

  FermionicOperator get_folded_operator(const int num_inactive_orbs,
                                        const int num_active_orbs,
                                        const int num_virtual_orbs) {
    std::map<OperatorString, double> folded_operators;
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
      OperatorString virtual_;
      OperatorString virtual_dagger;
      OperatorString inactive;
      OperatorString inactive_dagger;
      OperatorString active;
      OperatorString active_dagger;
      double fac = 1;

      for (auto anni : ops) {
        int idx = std::get<0>(anni);
        bool dagger = std::get<1>(anni);
        if (dagger) {
          if (std::find(inactive_idx.begin(), inactive_idx.end(), idx) !=
              inactive_idx.end()) {
            inactive_dagger.push_back(anni);
          } else if (std::find(active_idx.begin(), active_idx.end(), idx) !=
                     active_idx.end()) {
            active_dagger.push_back(anni);
          } else {
            virtual_dagger.push_back(anni);
          }
        } else {
          if (std::find(inactive_idx.begin(), inactive_idx.end(), idx) !=
              inactive_idx.end()) {
            inactive.push_back(anni);
          } else if (std::find(active_idx.begin(), active_idx.end(), idx) !=
                     active_idx.end()) {
            active.push_back(anni);
          } else {
            virtual_.push_back(anni);
          }
        }
      }
      if (virtual_.size() != 0 || virtual_dagger.size() != 0) {
        continue;
      }
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
      for (int i = 1; i < (int)ket_side.size() + 1; i++) {
        if (i % 2 == 0) {
          ket_flip_fac *= -1;
        }
      }
      fac *= ket_flip_fac;
      const auto new_key = active_op;
      folded_operators[new_key] += fac * operators[ops];
      if (std::abs(folded_operators[new_key]) < 1e-14) {
        folded_operators.erase(new_key);
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