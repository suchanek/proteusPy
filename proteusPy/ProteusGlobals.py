#
# global directories for PDB files and the correspondig .pkl models
# this is relative to the proteusPy package folder hierarchy!
#

PDB_DIR = '../pdb/'
MODEL_DIR = f'{PDB_DIR}models/'
DATA_DIR = '../data/'

# global for initialization of dihedrals and energies

ORIENT_BACKBONE = 2
ORIENT_SIDECHAIN = 1

# Cα Cβ Sγ Χ1 - Χ5 Χ

#/* atoms_covalent.inc - covalent radii used by atoms.inc */
#/* from Pauling and CAChe S/W */
# /* 5/4/94 -egs- */

N_RAD_COV = 0.74
C_RAD_COV = 0.77
O_RAD_COV = 0.73
P_RAD_COV = 1.10
S_RAD_COV = 1.04
CA_RAD_COV = 1.74
H_RAD_COV = .35
FE_RAD_COV = 1.17

ATOM_RADII_COVALENT = {"N": N_RAD_COV, "C": C_RAD_COV, "CA": CA_RAD_COV, "O": O_RAD_COV, "SG": S_RAD_COV, "SG": S_RAD_COV, "H": H_RAD_COV, "CB": C_RAD_COV}

# /* atoms_cpk.inc - CPK radii for atoms, used by atoms.inc */

N_RAD_CPK = 1.54
C_RAD_CPK = 1.7
O_RAD_CPK = 1.4
P_RAD_CPK = 1.9
S_RAD_CPK = 1.8
CA_RAD_CPK = 1.274
H_RAD_CPK = 1.2

ATOM_RADII_CPK = {"N": N_RAD_CPK, "C": C_RAD_CPK, "CA": CA_RAD_CPK, "O": O_RAD_CPK, "SG": S_RAD_CPK, "H": H_RAD_CPK, "CB": C_RAD_CPK}

BOND_RADIUS = .25


