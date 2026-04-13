# type: ignore
import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6
import sys


import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.sa_adapt_wavefunction5 import WaveFunctionSAADAPT

#from slowquant.unitary_coupled_cluster.ups_wavefunction import WaveFunctionUPS
#from slowquant.unitary_coupled_cluster.sa_ups_wavefunction import WaveFunctionSAUPS

"""This should give exactly the same as FCI.

Since all states, are includes in the subspace expansion.
"""
from pyscf import gto, scf ,mcscf
import numpy as np
np.set_printoptions(threshold=sys.maxsize,linewidth=100000,precision=4)
atm = """
Co       0.0002196495   -0.0004158533    0.0003260168;
O        2.0499167005    0.0041918004   -0.0307035275;
H        2.6381106076    0.0346982951    0.7181433118;
H        2.5772955933   -0.0261783595   -0.8237029335;
O       -2.0494474209   -0.0049730225    0.0314977781;
H       -2.5763157315    0.0259959000    0.8248106836;
H       -2.6381349142   -0.0368323414   -0.7168992052;
O       -0.0373575554    2.0256387008   -0.0574973678;
H        0.7295690941    2.5889202578   -0.0887301049;
H       -0.8103203640    2.5815520095   -0.0618723047;
O        0.0381845119   -2.0263348811    0.0569140121;
H        0.8112882514   -2.5820596513    0.0600548132;
H       -0.7284969339   -2.5899574225    0.0881318281;
O       -0.0936101874    0.0876342958    2.4075973890;
H       -0.0377190958    0.8604070567    2.9586351194;
H       -0.1319968785   -0.6459484541    3.0112267816;
O        0.0919168741   -0.0863094466   -2.4077514651;
H        0.1310414455    0.6481839596   -3.0102036573;
H        0.0358573372   -0.8582108355   -2.9599761709;
"""
#    """H   0.0  0.0  0.0;
#       Li   0.0  0.0  0.735;""",
#    molecular_charge=2,
#    distance_unit="angstrom",
#)

mol = gto.M(
    atom=atm,
    charge= 1,
    spin = 0,
    verbose=4,
    unit="Angstrom",
    basis= "def2-svp"
)


mf = scf.RHF(mol)
mf.max_cycle = 300
mf.kernel()

ml = mcscf.CASSCF(mf, 8, 8)
ml.max_cycle_macro = 100
ml.kernel()

print(mol.nelec)
print(mf.e_tot - mf.energy_nuc())
print(ml.e_tot - mf.energy_nuc())
lc = [[1.0],[1.0],[1.0],[1.0]]
ld = [["0011"],["0110"],["1001"],["1100"]]




c=[
            
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
                
            ]
d =              [
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


ikl = [
                ["1111000011110000"],
                ["1111000110110000"],
                ["1111000111100000"],
                ["1111010010110000"],
                ["1111010011100000"],
                ["1111010110100000"],
                ["1111001001110000"],
                ["1111001100110000"],
                ["1111001101100000"],
                ["1111011000110000"],
                ["1111011001100000"],
                ["1111011100100000"],
                ["1111001011010000"],
                ["1111001110010000"],
                ["1111001111000000"],
                ["1111011010010000"],
                ["1111011011000000"],
                ["1111011110000000"],
                ["1111100001110000"],
                ["1111100100110000"],
                ["1111100101100000"],
                ["1111110000110000"],
                ["1111110001100000"],
                ["1111110100100000"],
                ["1111100011010000"],
                ["1111100110010000"],
                ["1111100111000000"],
                ["1111110010010000"],
                ["1111110011000000"],
                ["1111110110000000"],
                ["1111101001010000"],
                ["1111101100010000"],
                ["1111101101000000"],
                ["1111111000010000"],
                ["1111111001000000"],
                ["1111111100000000"],
            ]


ikl2 = [
                ["110000111100"],
                ["110001101100"],
                ["110001111000"],
                ["110100101100"],
                ["110100111000"],
                ["110101101000"],
                ["110010011100"],
                ["110011001100"],
                ["110011011000"],
                ["110110001100"],
                ["110110011000"],
                ["110111001000"],
                ["110010110100"],
                ["110011100100"],
                ["110011110000"],
                ["110110100100"],
                ["110110110000"],
                ["110111100000"],
                ["111000011100"],
                ["111001001100"],
                ["111001011000"],
                ["111100001100"],
                ["111100011000"],
                ["111101001000"],
                ["111000110100"],
                ["111001100100"],
                ["111001110000"],
                ["111100100100"],
                ["111100110000"],
                ["111101100000"],
                ["111010010100"],
                ["111011000100"],
                ["111011010000"],
                ["111110000100"],
                ["111110010000"],
                ["111111000000"],
            ]

#with open('test.npy', 'rb') as f:
#    mco = np.load(f)
mco = ml.mo_coeff
WF = WaveFunctionSAADAPT(
#WF = WaveFunctionUPS(
    mol.nelec[0] + mol.nelec[1],
    (8, 8),
    mco,
    #mf.mo_coeff,
    mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
    mol.intor("int2e"),
    (
        c,ikl),
    "ADAPT",
    target_spin =  0,
    unpaired_electron = 0,
    spinfactor=0.001,
    state_specific=True
)



exc = []
exc_type = []

#WF.do_adapt(orbital_opt=False, epoch=1e-4, optimiser_algo="l-bfgs-b")

def solver(mol = None , mo = None, wfp = None):
    llg = SlowQuantADAPT_solver(mol,mo, wf=wfp)
    llg.kernel(None,None,None,None,None,None)
    return llg

class SlowQuantADAPT_solver:
    def __init__(self, mol, m_co=None, wf=None):
        self.WF = wf
        self.mm = mol
        self.eci = 0
        if(m_co.any() != None):
            self.WF.c_mo = m_co
            self.WF.do_adapt(orbital_opt=False, epoch=1e-4, optimiser_algo="l-bfgs-b")
        self.ci = self.WF.ci_coeffs
        if(m_co.any() == None):
            self.mo = []
        else:
            self.mo = m_co
        self.rewire= False
        self.store_exc = False
        self.static_ansatz = 0

        pass

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        print(self)
        print(self.rewire)
        global exc, exc_type
        if(len(self.mo) != 0):
            self.WF.c_mo = self.mo
        else:
            print("Mo None")
        
        if(self.rewire == True):
            
            self.WF.ups_layout.excitation_indices = []
            self.WF.ups_layout.excitation_operator_type = []
            self.WF.ups_layout.n_params = 0
            self.WF.thetas = []
            self.rewire = False
            with open("sa_adapt_wavefunction.txt", "a") as fwrite_f1:
                fwrite_f1.write(f"brite-----------------------------------------------brite\n")
            for i in range(len(exc)):
                self.WF.ups_layout.excitation_indices.append(np.array(exc[i]) - self.WF.num_inactive_spin_orbs)
                self.WF.ups_layout.excitation_operator_type.append(exc_type[i])
                self.WF.ups_layout.n_params += 1
                g = [0 for i in range(len(exc))]
                self.WF.thetas = g 
            print(self.WF.sa_energy)
            self.WF.do_adapt(orbital_opt=False, epoch=1e-4, optimiser_algo="l-bfgs-b")
            self.store_exc = True 
        print(self.WF.sa_energy)
        return self.WF.sa_energy, self.WF.ci_coeffs

    def make_rdm12(self,fcivec, ncas,nelecas):
        return self.WF.rdm1 , self.WF.rdm2



mc = mcscf.CASSCF(mf, 8, 8)
print("finc")
mc.fcisolver = solver(mc, WF.c_mo, wfp=WF)
mc.max_cycle_macro = 100
mc.kernel(mo_coeff=WF.c_mo)


with open('test.npy', 'wb') as f:
    np.save(f,mc.mo_coeff)
