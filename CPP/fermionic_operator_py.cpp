#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <sstream>
#include <iomanip>

#include "fermionic_operator.hpp"  // Assume the C++ implementation is in this header

namespace py = pybind11;
using namespace pybind11::literals;

// Helper function to convert Python tuples to OperatorString
OperatorString tuple_to_operator_string(const py::tuple& t) {
    OperatorString result;
    for (const auto& item : t) {
        auto pair = item.cast<py::tuple>();
        if (pair.size() != 2) {
            throw std::runtime_error("Each operator component must be a tuple of (int, bool)");
        }
        int idx = pair[0].cast<int>();
        bool is_dagger = pair[1].cast<bool>();
        result.emplace_back(idx, is_dagger);
    }
    return result;
}

// Helper function to convert OperatorString to Python tuple
py::tuple operator_string_to_tuple(const OperatorString& op_string) {
    py::list result;
    for (const auto& comp : op_string) {
        result.append(py::make_tuple(comp.first, comp.second));
    }
    return py::tuple(result);
}

// Helper to convert Python dict to internal operator map
std::map<OperatorString, double, OperatorStringCompare> 
py_dict_to_operator_map(const py::dict& dict) {
    std::map<OperatorString, double, OperatorStringCompare> result;
    for (const auto& item : dict) {
        py::tuple key_tuple = item.first.cast<py::tuple>();
        double value = item.second.cast<double>();
        result[tuple_to_operator_string(key_tuple)] = value;
    }
    return result;
}

// Helper to convert internal operator map to Python dict
py::dict operator_map_to_py_dict(const std::map<OperatorString, double, OperatorStringCompare>& op_map) {
    py::dict result;
    for (const auto& [op_string, value] : op_map) {
        result[operator_string_to_tuple(op_string)] = value;
    }
    return result;
}

PYBIND11_MODULE(fermionic_operator, m) {
    m.doc() = "FermionicOperator C++ implementation with pybind11 bindings";
    
    // Register the custom exception for type errors
    py::register_exception<py::type_error>(m, "TypeError");

    // Bind the main FermionicOperator class
    py::class_<FermionicOperator>(m, "FermionicOperator")
        // Constructors
        .def(py::init<>(), "Construct an empty FermionicOperator")
        .def(py::init<const std::map<OperatorString, double, OperatorStringCompare>&>(),
             "Construct from operator map")
        .def(py::init([](const py::dict& operators) {
                return FermionicOperator(py_dict_to_operator_map(operators));
             }), "Construct from Python dictionary of operators")
        
        // Properties
        .def_property_readonly("operators", [](const FermionicOperator& self) {
                return operator_map_to_py_dict(self.get_operators());
             }, "Get the operators dictionary")
        
        .def_property_readonly("operators_readable", &FermionicOperator::operators_readable,
             "Get the operator in human readable format")
        
        .def_property_readonly("operator_count", &FermionicOperator::operator_count,
             "Count number of operators of different lengths")
        
        .def_property_readonly("dagger", &FermionicOperator::dagger,
             "Get the Hermitian conjugate of the operator")
        
        // Magic methods
        .def(py::self + py::self, "Addition of two fermionic operators")
        .def(py::self += py::self, "In-place addition")
        
        .def(py::self - py::self, "Subtraction of two fermionic operators")
        .def(py::self -= py::self, "In-place subtraction")
        
        .def(py::self * py::self, "Multiplication of two fermionic operators")
        .def(py::self *= py::self, "In-place multiplication")
        
        .def(py::self * double(), "Multiply by scalar")
        .def(double() * py::self, "Multiply by scalar (left)")
        .def(py::self *= double(), "In-place scalar multiplication")
        
        .def(-py::self, "Negate the operator")
        
        // String representation
        .def("__repr__", [](const FermionicOperator& self) {
                std::stringstream ss;
                ss << "FermionicOperator(";
                auto readable = self.operators_readable();
                if (readable.empty()) {
                    ss << "empty";
                } else {
                    bool first = true;
                    for (const auto& [op_str, coeff] : readable) {
                        if (!first) ss << ", ";
                        ss << op_str << ": " << std::fixed << std::setprecision(6) << coeff;
                        first = false;
                    }
                }
                ss << ")";
                return ss.str();
             })
        
        .def("__str__", [](const FermionicOperator& self) {
                std::stringstream ss;
                auto readable = self.operators_readable();
                if (readable.empty()) {
                    ss << "0";
                } else {
                    bool first = true;
                    for (const auto& [op_str, coeff] : readable) {
                        if (!first) ss << " + ";
                        ss << coeff << " * " << op_str;
                        first = false;
                    }
                }
                return ss.str();
             })
        
        // Comparison operators
        .def(py::self == py::self)
        .def(py::self != py::self)
        
        // Main methods
        .def("get_qiskit_form", &FermionicOperator::get_qiskit_form, 
             "num_orbs"_a, "Get fermionic operator in qiskit form")
        
        .def("get_folded_operator", &FermionicOperator::get_folded_operator,
             "num_inactive_orbs"_a, "num_active_orbs"_a, "num_virtual_orbs"_a,
             "Get folded operator for active space methods")
        
        .def("get_info", &FermionicOperator::get_info,
             "Return operator excitation in ordered strings with coefficient")
        
        .def("normal_order", [](const FermionicOperator& self) {
                return FermionicOperator(do_extended_normal_ordering(self));
             }, "Return normal-ordered version of the operator")
        
        // Check if operator is zero
        .def("is_zero", [](const FermionicOperator& self) {
                return self.get_operators().empty();
             })
        
        // Get number of terms
        .def("__len__", [](const FermionicOperator& self) {
                return self.get_operators().size();
             })
        
        // Iterate over terms (Python iteration support)
        .def("__iter__", [](const FermionicOperator& self) {
                auto ops = self.operators_readable();
                return py::make_iterator(ops.begin(), ops.end());
             }, py::keep_alive<0, 1>());

    // Bind the free function do_extended_normal_ordering
    m.def("do_extended_normal_ordering", [](const FermionicOperator& op) {
            return FermionicOperator(do_extended_normal_ordering(op));
         }, "op"_a, "Reorder fermionic operator string (normal ordering)");

    m.def("operator_to_qiskit_key", &operator_to_qiskit_key, 
          "operator_string"_a, "remapping"_a, 
          "Make key string to index a fermionic operator");

    // Add version information
    m.attr("__version__") = "1.0.0";

    // Add some useful constants
    m.attr("ZERO_TOLERANCE") = 1e-14;

    // Create a submodule for utility functions
    py::module utils = m.def_submodule("utils", "Utility functions for fermionic operators");
    
    utils.def("create_creation", [](int idx, double coeff) {
            OperatorString op = {{idx, true}};
            return FermionicOperator({{op, coeff}});
         }, "idx"_a, "coeff"_a = 1.0, "Create a single creation operator");
    
    utils.def("create_annihilation", [](int idx, double coeff) {
            OperatorString op = {{idx, false}};
            return FermionicOperator({{op, coeff}});
         }, "idx"_a, "coeff"_a = 1.0, "Create a single annihilation operator");
    
    utils.def("create_number_operator", [](int idx, double coeff) {
            OperatorString op = {{idx, true}, {idx, false}};
            return FermionicOperator({{op, coeff}});
         }, "idx"_a, "coeff"_a = 1.0, "Create a number operator n_i = c_i† c_i");
    
    utils.def("create_hopping", [](int i, int j, double coeff) {
            OperatorString op1 = {{i, true}, {j, false}};
            OperatorString op2 = {{j, true}, {i, false}};
            std::map<OperatorString, double, OperatorStringCompare> ops;
            ops[op1] = coeff;
            ops[op2] = coeff;
            return FermionicOperator(ops);
         }, "i"_a, "j"_a, "coeff"_a = 1.0, 
         "Create hopping operator t (c_i† c_j + c_j† c_i)");
    
    utils.def("create_interaction", [](int i, int j, double coeff) {
            OperatorString op = {{i, true}, {j, true}, {j, false}, {i, false}};
            return FermionicOperator({{op, coeff}});
         }, "i"_a, "j"_a, "coeff"_a = 1.0, 
         "Create interaction operator V n_i n_j");
    
    utils.def("create_pairing", [](int i, int j, double coeff) {
            OperatorString op1 = {{i, true}, {j, true}};
            OperatorString op2 = {{j, false}, {i, false}};
            std::map<OperatorString, double, OperatorStringCompare> ops;
            ops[op1] = coeff;
            ops[op2] = coeff;
            return FermionicOperator(ops);
         }, "i"_a, "j"_a, "coeff"_a = 1.0, 
         "Create pairing operator Δ (c_i† c_j† + c_j c_i)");

    // Add a convenience function to create Hamiltonian
    utils.def("create_hubbard_hamiltonian", [](int num_sites, double t, double U, double mu) {
            std::map<OperatorString, double, OperatorStringCompare> hamiltonian;
            
            // Hopping term: -t ∑_{i,σ} (c_{i,σ}† c_{i+1,σ} + h.c.)
            for (int i = 0; i < num_sites - 1; ++i) {
                for (int sigma = 0; sigma < 2; ++sigma) {
                    int idx1 = 2 * i + sigma;
                    int idx2 = 2 * (i + 1) + sigma;
                    
                    OperatorString hop1 = {{idx1, true}, {idx2, false}};
                    OperatorString hop2 = {{idx2, true}, {idx1, false}};
                    
                    hamiltonian[hop1] += -t;
                    hamiltonian[hop2] += -t;
                }
            }
            
            // Interaction term: U ∑_i n_{i,↑} n_{i,↓}
            for (int i = 0; i < num_sites; ++i) {
                int up = 2 * i;
                int down = 2 * i + 1;
                OperatorString interaction = {{up, true}, {down, true}, {down, false}, {up, false}};
                hamiltonian[interaction] += U;
            }
            
            // Chemical potential: -μ ∑_{i,σ} n_{i,σ}
            if (std::abs(mu) > 1e-14) {
                for (int i = 0; i < num_sites; ++i) {
                    for (int sigma = 0; sigma < 2; ++sigma) {
                        int idx = 2 * i + sigma;
                        OperatorString number = {{idx, true}, {idx, false}};
                        hamiltonian[number] += -mu;
                    }
                }
            }
            
            return FermionicOperator(hamiltonian);
         }, "num_sites"_a, "t"_a = 1.0, "U"_a = 0.0, "mu"_a = 0.0,
         "Create a 1D Hubbard model Hamiltonian");
}