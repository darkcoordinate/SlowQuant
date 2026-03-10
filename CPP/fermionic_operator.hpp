#include <vector>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>
#include <regex>
#include <memory>
#include <iostream>
#include <utility>

// Type aliases for fermionic operator representation
using SpinOrbitalIndex = int;
using IsDagger = bool;
using OperatorComponent = std::pair<SpinOrbitalIndex, IsDagger>;
using OperatorString = std::vector<OperatorComponent>;
using OperatorKey = std::vector<OperatorComponent>;  // For map keys we'll use string representation

// Helper to convert OperatorString to a string key for maps
std::string operator_string_to_key(const OperatorString& op_string) {
    std::stringstream ss;
    for (const auto& comp : op_string) {
        ss << comp.first << (comp.second ? "_d" : "_a") << "_";
    }
    return ss.str();
}

// Helper to parse key back to OperatorString
OperatorString key_to_operator_string(const std::string& key) {
    OperatorString result;
    std::stringstream ss(key);
    std::string item;
    while (std::getline(ss, item, '_')) {
        if (item.empty()) continue;
        size_t pos = item.find('_');
        if (pos != std::string::npos) {
            int idx = std::stoi(item.substr(0, pos));
            bool is_dagger = (item.substr(pos + 1) == "d");
            result.emplace_back(idx, is_dagger);
        }
    }
    return result;
}

// Forward declaration
class FermionicOperator;

std::string operator_to_qiskit_key(const OperatorString& operator_string, 
                                   const std::map<int, int>& remapping) {
    std::string op_key;
    for (const auto& a : operator_string) {
        if (a.second) {
            op_key += " +_" + std::to_string(remapping.at(a.first));
        } else {
            op_key += " -_" + std::to_string(remapping.at(a.first));
        }
    }
    // Remove leading space if present
    if (!op_key.empty() && op_key[0] == ' ') {
        return op_key.substr(1);
    }
    return op_key;
}

std::map<OperatorString, double, std::function<bool(const OperatorString&, const OperatorString&)>> 
do_extended_normal_ordering(const FermionicOperator& fermistring);

// Custom comparator for OperatorString to use as map key
struct OperatorStringCompare {
    bool operator()(const OperatorString& lhs, const OperatorString& rhs) const {
        if (lhs.size() != rhs.size()) return lhs.size() < rhs.size();
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i].first != rhs[i].first) return lhs[i].first < rhs[i].first;
            if (lhs[i].second != rhs[i].second) return lhs[i].second < rhs[i].second;
        }
        return false;
    }
};

class FermionicOperator {
private:
    std::map<OperatorString, double, OperatorStringCompare> operators;

public:
    // Constructor
    FermionicOperator() = default;
    
    explicit FermionicOperator(const std::map<OperatorString, double, OperatorStringCompare>& ops) 
        : operators(ops) {}

    // Accessors
    const auto& get_operators() const { return operators; }
    auto& get_operators() { return operators; }

    // Addition
    FermionicOperator operator+(const FermionicOperator& other) const {
        FermionicOperator result(*this);
        result += other;
        return result;
    }

    FermionicOperator& operator+=(const FermionicOperator& other) {
        for (const auto& [op_key, value] : other.operators) {
            auto it = operators.find(op_key);
            if (it != operators.end()) {
                it->second += value;
                if (std::abs(it->second) < 1e-14) {
                    operators.erase(it);
                }
            } else {
                operators[op_key] = value;
            }
        }
        return *this;
    }

    // Subtraction
    FermionicOperator operator-(const FermionicOperator& other) const {
        FermionicOperator result(*this);
        result -= other;
        return result;
    }

    FermionicOperator& operator-=(const FermionicOperator& other) {
        for (const auto& [op_key, value] : other.operators) {
            auto it = operators.find(op_key);
            if (it != operators.end()) {
                it->second -= value;
                if (std::abs(it->second) < 1e-14) {
                    operators.erase(it);
                }
            } else {
                operators[op_key] = -value;
            }
        }
        return *this;
    }

    // Multiplication (by scalar or FermionicOperator)
    FermionicOperator operator*(double scalar) const {
        FermionicOperator result(*this);
        result *= scalar;
        return result;
    }

    FermionicOperator operator*(const FermionicOperator& other) const {
        std::map<OperatorString, double, OperatorStringCompare> new_ops;
        
        for (const auto& [op_key1, val1] : other.operators) {
            for (const auto& [op_key2, val2] : this->operators) {
                // Combine strings: op_key2 + op_key1
                OperatorString combined = op_key2;
                combined.insert(combined.end(), op_key1.begin(), op_key1.end());
                
                FermionicOperator temp({{combined, val2 * val1}});
                auto ordered_ops = do_extended_normal_ordering(temp);
                
                for (const auto& [new_key, new_val] : ordered_ops) {
                    auto it = new_ops.find(new_key);
                    if (it != new_ops.end()) {
                        it->second += new_val;
                        if (std::abs(it->second) < 1e-14) {
                            new_ops.erase(it);
                        }
                    } else {
                        new_ops[new_key] = new_val;
                    }
                }
            }
        }
        return FermionicOperator(new_ops);
    }

    FermionicOperator& operator*=(double scalar) {
        for (auto& [op_key, value] : operators) {
            value *= scalar;
        }
        return *this;
    }

    FermionicOperator& operator*=(const FermionicOperator& other) {
        auto result = (*this) * other;
        operators = std::move(result.operators);
        return *this;
    }

    // Friend for scalar * FermionicOperator
    friend FermionicOperator operator*(double scalar, const FermionicOperator& op) {
        return op * scalar;
    }

    // Negation
    FermionicOperator operator-() const {
        FermionicOperator result(*this);
        for (auto& [op_key, value] : result.operators) {
            value = -value;
        }
        return result;
    }

    // Dagger operation
    FermionicOperator dagger() const {
        std::map<OperatorString, double, OperatorStringCompare> new_ops;
        
        for (const auto& [op_key, value] : operators) {
            OperatorString new_op;
            for (auto it = op_key.rbegin(); it != op_key.rend(); ++it) {
                new_op.emplace_back(it->first, !it->second);
            }
            new_ops[new_op] = value;
        }
        
        auto ordered_ops = do_extended_normal_ordering(FermionicOperator(new_ops));
        return FermionicOperator(ordered_ops);
    }

    // Count operators by length
    std::map<int, int> operator_count() const {
        std::map<int, int> counts;
        for (const auto& [op_key, value] : operators) {
            int len = static_cast<int>(op_key.size());
            counts[len]++;
        }
        return counts;
    }

    // Human readable format
    std::map<std::string, double> operators_readable() const {
        std::map<std::string, double> readable;
        for (const auto& [op_string, value] : operators) {
            std::stringstream ss;
            for (const auto& comp : op_string) {
                if (comp.second) {
                    ss << "c" << comp.first;
                } else {
                    ss << "a" << comp.first;
                }
            }
            readable[ss.str()] = value;
        }
        return readable;
    }

    // Qiskit format
    std::map<std::string, double> get_qiskit_form(int num_orbs) const {
        std::map<std::string, double> qiskit_form;
        std::map<int, int> remapping;
        
        for (int i = 0; i < 2 * num_orbs; ++i) {
            if (i < num_orbs) {
                remapping[2 * i] = i;
            } else {
                remapping[2 * i + 1 - 2 * num_orbs] = i;
            }
        }
        
        for (const auto& [op_key, value] : operators) {
            std::string qiskit_str = operator_to_qiskit_key(op_key, remapping);
            qiskit_form[qiskit_str] = value;
        }
        return qiskit_form;
    }

    // Folded operator
    FermionicOperator get_folded_operator(int num_inactive_orbs, int num_active_orbs, 
                                          int num_virtual_orbs) const {
        std::map<OperatorString, double, OperatorStringCompare> result_ops;
        
        std::set<int> inactive_idx, active_idx, virtual_idx;
        int total_orbs = 2 * (num_inactive_orbs + num_active_orbs + num_virtual_orbs);
        
        for (int i = 0; i < total_orbs; ++i) {
            if (i < 2 * num_inactive_orbs) {
                inactive_idx.insert(i);
            } else if (i < 2 * (num_inactive_orbs + num_active_orbs)) {
                active_idx.insert(i);
            } else {
                virtual_idx.insert(i);
            }
        }
        
        for (const auto& [op_key, value] : operators) {
            std::vector<int> virtual_ops, virtual_dagger, inactive, inactive_dagger;
            OperatorString active_ops;
            double fac = 1.0;
            
            for (const auto& anni : op_key) {
                if (anni.second) { // dagger
                    if (inactive_idx.count(anni.first)) {
                        inactive_dagger.push_back(anni.first);
                    } else if (active_idx.count(anni.first)) {
                        active_ops.emplace_back(anni.first - 2 * num_inactive_orbs, true);
                    } else if (virtual_idx.count(anni.first)) {
                        virtual_dagger.push_back(anni.first);
                    }
                } else { // not dagger
                    if (inactive_idx.count(anni.first)) {
                        inactive.push_back(anni.first);
                    } else if (active_idx.count(anni.first)) {
                        active_ops.emplace_back(anni.first - 2 * num_inactive_orbs, false);
                    } else if (virtual_idx.count(anni.first)) {
                        virtual_ops.push_back(anni.first);
                    }
                }
            }
            
            // Skip if any virtual indices
            if (!virtual_ops.empty() || !virtual_dagger.empty()) {
                continue;
            }
            
            // Check if bra and ket sides match
            if (inactive_dagger != inactive) {
                continue;
            }
            
            if (inactive_dagger.size() % 2 == 1 && active_ops.size() % 2 == 1) {
                fac *= -1;
            }
            
            // Calculate ket flip factor
            double ket_flip_fac = 1.0;
            for (size_t i = 1; i <= inactive.size(); ++i) {
                if (i % 2 == 0) {
                    ket_flip_fac *= -1;
                }
            }
            fac *= ket_flip_fac;
            
            auto it = result_ops.find(active_ops);
            if (it != result_ops.end()) {
                it->second += fac * value;
                if (std::abs(it->second) < 1e-14) {
                    result_ops.erase(it);
                }
            } else {
                result_ops[active_ops] = fac * value;
            }
        }
        
        return FermionicOperator(result_ops);
    }

    // Get info: annihilation, creation, coefficients
    std::tuple<std::vector<std::vector<int>>, 
               std::vector<std::vector<int>>, 
               std::vector<double>> get_info() const {
        auto readable = operators_readable();
        std::vector<std::vector<int>> annihilation;
        std::vector<std::vector<int>> creation;
        std::vector<double> coefficients;
        
        std::regex number_regex("\\d+");
        
        for (const auto& [op_string, coeff] : readable) {
            std::vector<int> numbers;
            auto words_begin = std::sregex_iterator(op_string.begin(), op_string.end(), number_regex);
            auto words_end = std::sregex_iterator();
            
            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                numbers.push_back(std::stoi((*i).str()));
            }
            
            size_t midpoint = numbers.size() / 2;
            std::vector<int> c(numbers.begin(), numbers.begin() + midpoint);
            std::vector<int> a(numbers.begin() + midpoint, numbers.end());
            
            creation.push_back(c);
            annihilation.push_back(a);
            coefficients.push_back(coeff);
        }
        
        return {annihilation, creation, coefficients};
    }
};

// Implementation of do_extended_normal_ordering
std::map<OperatorString, double, std::function<bool(const OperatorString&, const OperatorString&)>> 
do_extended_normal_ordering(const FermionicOperator& fermistring) {
    using OpMap = std::map<OperatorString, double, OperatorStringCompare>;
    
    std::vector<OperatorString> operator_queue;
    std::vector<double> factor_queue;
    OpMap new_operators;
    
    for (const auto& [key, value] : fermistring.get_operators()) {
        operator_queue.push_back(key);
        factor_queue.push_back(value);
    }
    
    while (!operator_queue.empty()) {
        auto next_operator = operator_queue.front();
        operator_queue.erase(operator_queue.begin());
        double factor = factor_queue.front();
        factor_queue.erase(factor_queue.begin());
        
        while (true) {
            bool changed = false;
            bool is_zero = false;
            
            for (size_t current_idx = 0; current_idx + 1 < next_operator.size(); ++current_idx) {
                const auto& a = next_operator[current_idx];
                const auto& b = next_operator[current_idx + 1];
                
                if (a.second && b.second) { // both dagger
                    if (a.first == b.first) {
                        is_zero = true;
                        break;
                    } else if (a.first < b.first) {
                        std::swap(next_operator[current_idx], next_operator[current_idx + 1]);
                        factor *= -1;
                        changed = true;
                    }
                } else if (!a.second && b.second) { // annihilation then dagger
                    if (a.first == b.first) {
                        auto new_op = next_operator;
                        new_op.erase(new_op.begin() + current_idx + 1);
                        new_op.erase(new_op.begin() + current_idx);
                        if (!new_op.empty()) {
                            operator_queue.push_back(new_op);
                            factor_queue.push_back(factor);
                        }
                        std::swap(next_operator[current_idx], next_operator[current_idx + 1]);
                        factor *= -1;
                        changed = true;
                    } else {
                        std::swap(next_operator[current_idx], next_operator[current_idx + 1]);
                        factor *= -1;
                        changed = true;
                    }
                } else if (a.second && !b.second) {
                    // dagger then annihilation - no change needed
                    continue;
                } else { // both annihilation
                    if (a.first == b.first) {
                        is_zero = true;
                        break;
                    } else if (a.first < b.first) {
                        std::swap(next_operator[current_idx], next_operator[current_idx + 1]);
                        factor *= -1;
                        changed = true;
                    }
                }
            }
            
            if (!changed || is_zero) {
                if (!is_zero) {
                    auto it = new_operators.find(next_operator);
                    if (it != new_operators.end()) {
                        it->second += factor;
                        if (std::abs(it->second) < 1e-14) {
                            new_operators.erase(it);
                        }
                    } else {
                        new_operators[next_operator] = factor;
                    }
                }
                break;
            }
        }
    }
    
    return new_operators;
}