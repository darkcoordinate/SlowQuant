#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include "fermionic_operator.hpp"

// Include the FermionicOperator class definition here...

void print_operator(const std::string& title, const FermionicOperator& op) {
    std::cout << "\n" << title << ":\n";
    std::cout << std::string(50, '-') << "\n";
    
    auto readable = op.operators_readable();
    if (readable.empty()) {
        std::cout << "  (empty operator)\n";
    } else {
        for (const auto& [op_str, coeff] : readable) {
            std::cout << "  " << std::setw(15) << op_str << " : " 
                      << std::setw(10) << std::fixed << std::setprecision(6) << coeff << "\n";
        }
    }
}

void print_qiskit_format(const FermionicOperator& op, int num_orbs) {
    std::cout << "\nQiskit format (" << num_orbs << " orbitals):\n";
    std::cout << std::string(50, '-') << "\n";
    
    auto qiskit_form = op.get_qiskit_form(num_orbs);
    for (const auto& [qiskit_str, coeff] : qiskit_form) {
        std::cout << "  " << std::setw(20) << qiskit_str << " : " 
                  << std::setw(10) << std::fixed << std::setprecision(6) << coeff << "\n";
    }
}

void print_operator_counts(const FermionicOperator& op) {
    std::cout << "\nOperator counts by length:\n";
    std::cout << std::string(50, '-') << "\n";
    
    auto counts = op.operator_count();
    for (const auto& [length, count] : counts) {
        std::cout << "  Length " << length << ": " << count << " operators\n";
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "FermionicOperator C++ Implementation Test\n";
    std::cout << "========================================\n";
    
    // Test 1: Create basic operators
    std::cout << "\nTEST 1: Creating basic operators\n";
    
    // Create single creation operator: c0†
    OperatorString c0_dagger = {{0, true}};
    FermionicOperator op1({{c0_dagger, 1.0}});
    print_operator("Creation operator c0†", op1);
    
    // Create single annihilation operator: a0
    OperatorString a0 = {{0, false}};
    FermionicOperator op2({{a0, 1.0}});
    print_operator("Annihilation operator a0", op2);
    
    // Create a two-body operator: c1† c0† a0 a1 (excitation operator)
    OperatorString two_body = {{1, true}, {0, true}, {0, false}, {1, false}};
    FermionicOperator op3({{two_body, 2.5}});
    print_operator("Two-body operator (coefficient 2.5)", op3);
    
    // Test 2: Operator algebra
    std::cout << "\n\nTEST 2: Operator algebra\n";
    
    // Addition
    std::cout << "\n--- Addition ---\n";
    auto sum_op = op1 + op2;
    print_operator("op1 + op2", sum_op);
    
    // Subtraction
    std::cout << "\n--- Subtraction ---\n";
    auto diff_op = op1 - op2;
    print_operator("op1 - op2", diff_op);
    
    // Scalar multiplication
    std::cout << "\n--- Scalar multiplication ---\n";
    auto scalar_op = 3.0 * op3;
    print_operator("3.0 * op3", scalar_op);
    
    // Test 3: Operator multiplication (anticommutation relations)
    std::cout << "\n\nTEST 3: Multiplication and anticommutation\n";
    
    // Test anticommutation: {c_i†, c_j} = δ_ij
    std::cout << "\n--- Anticommutation: {c0†, c0} ---\n";
    auto anticomm_1 = op1 * op2;  // c0† * a0
    auto anticomm_2 = op2 * op1;  // a0 * c0†
    auto anticomm_sum = anticomm_1 + anticomm_2;  // Should be 1.0
    
    print_operator("c0† * a0", anticomm_1);
    print_operator("a0 * c0†", anticomm_2);
    print_operator("Sum (should be 1.0)", anticomm_sum);
    
    // Test different indices: {c0†, c1} = 0
    std::cout << "\n--- Anticommutation: {c0†, c1} ---\n";
    OperatorString c1 = {{1, false}};
    FermionicOperator op_c1({{c1, 1.0}});
    
    auto anticomm_3 = op1 * op_c1;  // c0† * a1
    auto anticomm_4 = op_c1 * op1;  // a1 * c0†
    auto anticomm_sum2 = anticomm_3 + anticomm_4;  // Should be 0
    
    print_operator("c0† * a1", anticomm_3);
    print_operator("a1 * c0†", anticomm_4);
    print_operator("Sum (should be 0)", anticomm_sum2);
    
    // Test 4: Normal ordering
    std::cout << "\n\nTEST 4: Normal ordering\n";
    
    // Create a non-normal ordered operator: a0 c0† (should become 1 - c0† a0)
    OperatorString non_normal = {{0, false}, {0, true}};
    FermionicOperator op4({{non_normal, 1.0}});
    print_operator("Before normal ordering: a0 c0†", op4);
    
    auto normal_ordered = do_extended_normal_ordering(op4);
    FermionicOperator op4_normal(normal_ordered);
    print_operator("After normal ordering", op4_normal);
    
    // Test 5: Dagger operation
    std::cout << "\n\nTEST 5: Dagger operation\n";
    
    print_operator("Original operator", op3);
    auto dag_op = op3.dagger();
    print_operator("Dagger of operator", dag_op);
    
    // Test 6: Qiskit format
    std::cout << "\n\nTEST 6: Qiskit format conversion\n";
    
    // Create a more complex operator with multiple terms
    OperatorString term1 = {{0, true}, {1, false}};  // c0† a1
    OperatorString term2 = {{1, true}, {0, false}};  // c1† a0
    OperatorString term3 = {{0, true}, {1, true}, {1, false}, {0, false}};  // c0† c1† a1 a0
    
    std::map<OperatorString, double, OperatorStringCompare> complex_ops;
    complex_ops[term1] = 1.0;
    complex_ops[term2] = 2.0;
    complex_ops[term3] = 0.5;
    
    FermionicOperator complex_op(complex_ops);
    print_operator("Complex operator", complex_op);
    print_qiskit_format(complex_op, 2);  // 2 spatial orbitals
    
    // Test 7: Operator counts
    std::cout << "\n\nTEST 7: Operator counting\n";
    print_operator_counts(complex_op);
    
    // Test 8: Folded operator (for active space methods)
    std::cout << "\n\nTEST 8: Folded operator (inactive=1, active=1, virtual=1)\n";
    
    // Create operator with indices spanning different spaces
    // Inactive indices: 0,1 (2 spin orbitals for 1 spatial)
    // Active indices: 2,3 (2 spin orbitals for 1 spatial)
    // Virtual indices: 4,5 (2 spin orbitals for 1 spatial)
    
    OperatorString full_space_op = {{2, true}, {3, true}, {3, false}, {2, false}};  // Active space only
    OperatorString mixed_op = {{0, true}, {2, false}};  // Inactive creation, active annihilation
    OperatorString virtual_op = {{4, true}, {5, false}};  // Virtual operator (should vanish)
    
    std::map<OperatorString, double, OperatorStringCompare> folding_test_ops;
    folding_test_ops[full_space_op] = 1.0;
    folding_test_ops[mixed_op] = 2.0;
    folding_test_ops[virtual_op] = 3.0;
    
    FermionicOperator folding_test(folding_test_ops);
    print_operator("Before folding", folding_test);
    
    auto folded = folding_test.get_folded_operator(1, 1, 1);  // 1 inactive, 1 active, 1 virtual
    print_operator("After folding (virtual terms should vanish)", folded);
    
    // Test 9: Get info (annihilation/creation lists)
    std::cout << "\n\nTEST 9: Getting excitation info\n";
    
    auto [annihilation, creation, coefficients] = complex_op.get_info();
    
    std::cout << "Number of terms: " << coefficients.size() << "\n";
    for (size_t i = 0; i < coefficients.size(); ++i) {
        std::cout << "  Term " << i + 1 << ":\n";
        std::cout << "    Creation: [";
        for (size_t j = 0; j < creation[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << creation[i][j];
        }
        std::cout << "]\n";
        
        std::cout << "    Annihilation: [";
        for (size_t j = 0; j < annihilation[i].size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << annihilation[i][j];
        }
        std::cout << "]\n";
        
        std::cout << "    Coefficient: " << coefficients[i] << "\n";
    }
    
    // Test 10: Complex algebra with multiple terms
    std::cout << "\n\nTEST 10: Complex algebra (Hamiltonian-like terms)\n";
    
    // Create a simple Hamiltonian: H = ε (c0† c0) + t (c0† c1 + c1† c0)
    OperatorString h1 = {{0, true}, {0, false}};  // n0 = c0† c0
    OperatorString h2 = {{0, true}, {1, false}};  // c0† c1
    OperatorString h3 = {{1, true}, {0, false}};  // c1† c0
    
    std::map<OperatorString, double, OperatorStringCompare> hamiltonian_ops;
    hamiltonian_ops[h1] = 1.5;  // ε = 1.5
    hamiltonian_ops[h2] = 0.5;  // t = 0.5
    hamiltonian_ops[h3] = 0.5;  // t = 0.5
    
    FermionicOperator H(hamiltonian_ops);
    print_operator("Hamiltonian H", H);
    
    // Compute H²
    std::cout << "\n--- Computing H² ---\n";
    auto H_squared = H * H;
    print_operator("H²", H_squared);
    print_operator_counts(H_squared);
    
    // Test commutator [H, n0] where n0 = c0† c0
    std::cout << "\n--- Computing commutator [H, n0] ---\n";
    FermionicOperator n0({{h1, 1.0}});
    auto commutator = H * n0 - n0 * H;
    print_operator("[H, n0]", commutator);
    
    std::cout << "\n========================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}