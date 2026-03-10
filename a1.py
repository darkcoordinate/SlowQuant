# type: ignore
import os
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import sys


import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.sa_adapt_wavefunction import WaveFunctionSAADAPT

#from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
#from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS

"""This should give exactly the same as FCI.

Since all states, are includes in the subspace expansion.
"""
from pyscf import gto, scf ,mcscf
import numpy as np
atm= """
C        0.0001201395   -0.0000565799   -0.0000000000;
O        1.2796857221    0.0000273114    0.0000000000;
O       -0.6398921477    1.1081454352   -0.0000000000;
O       -0.6399137140   -1.1081161667    0.0000000000;
"""


mol = gto.M(
    atom=atm,
    charge= -2,
    #charge= 0,
    spin = 0,
    verbose=4,
    unit="Angstrom",
    basis= "def2-svp"
)

mf = scf.RHF(mol)
mf.max_cycle = 300
mf.kernel()

print(mol.nelec)
print(mf.e_tot - mf.energy_nuc())

WF = WaveFunctionSAADAPT(
#WF = WaveFunctionUPS(
    mol.nelec[0] + mol.nelec[1],
    (4, 4),
    mf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    (
            [
            
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                
            ],
            [
                ["00001111"],
                ["00011011"],
                ["00011110"],
                ["01001011"],
                ["01001110"],
                ["01011010"],
                ["00100111"],
                ["00110011"],
                ["00110110"],
                ["01100011"],
                ["01100110"],
                ["01110010"],
                ["00101101"],
                ["00111001"],
                ["00111100"],
                ["01101001"],
                ["01101100"],
                ["01111000"],
                ["10000111"],
                ["10010011"],
                ["10010110"],
                ["11000011"],
                ["11000110"],
                ["11010010"],
                ["10001101"],
                ["10011001"],
                ["10011100"],
                ["11001001"],
                ["11001100"],
                ["11011000"],
                ["10100101"],
                ["10110001"],
                ["10110100"],
                ["11100001"],
                ["11100100"],
                ["11110000"],
            ],
    ),
    "ADAPT",
    target_spin =  0,
    unpaired_electron = 0,
    spinfactor=0.01,
    state_specific=True
)
ikl = [
                ["00001111"],
                ["00011011"],
                ["00011110"],
                ["01001011"],
                ["01001110"],
                ["01011010"],
                ["00100111"],
                ["00110011"],
                ["00110110"],
                ["01100011"],
                ["01100110"],
                ["01110010"],
                ["00101101"],
                ["00111001"],
                ["00111100"],
                ["01101001"],
                ["01101100"],
                ["01111000"],
                ["10000111"],
                ["10010011"],
                ["10010110"],
                ["11000011"],
                ["11000110"],
                ["11010010"],
                ["10001101"],
                ["10011001"],
                ["10011100"],
                ["11001001"],
                ["11001100"],
                ["11011000"],
                ["10100101"],
                ["10110001"],
                ["10110100"],
                ["11100001"],
                ["11100100"],
                ["11110000"],
            ]
print(len(ikl))



WF.do_adapt(orbital_opt=False, epoch=1e-5, optimiser_algo="l-bfgs-b")
print(WF.ci_coeffs)


print(WF.energy_states)
