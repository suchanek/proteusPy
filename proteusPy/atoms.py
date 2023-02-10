# atoms.py
# various radii used within the proteusPy program
# ancient vestiges of code shown below
#/* atoms.py - covalent radii used by atoms.inc */
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

ATOM_RADII_COVALENT = {"N": N_RAD_COV, "C": C_RAD_COV, "CA": CA_RAD_COV, 
                    "O": O_RAD_COV, "S": S_RAD_COV, "SG": S_RAD_COV, "H": H_RAD_COV, 
                    "CB": C_RAD_COV}

# /* atoms_cpk.inc - CPK radii for atoms, used by atoms.inc */

N_RAD_CPK = 1.54
C_RAD_CPK = 1.7
O_RAD_CPK = 1.4
P_RAD_CPK = 1.9
S_RAD_CPK = 1.8
CA_RAD_CPK = 1.274
H_RAD_CPK = 1.2
Z_RAD_CPK = .8

ATOM_RADII_CPK = {"N": N_RAD_CPK, "C": C_RAD_CPK, "CA": CA_RAD_CPK, 
                  "O": O_RAD_CPK, "SG": S_RAD_CPK, "S": S_RAD_CPK, 
                  "H": H_RAD_CPK, "CB": C_RAD_CPK, "Z": Z_RAD_CPK}
                  
ATOM_COLORS = {'O': 'red', 'C': 'grey', 'N': 'blue', 'S': 'yellow', 'H': 'white', 
               'SG': 'yellow', 'CB': 'grey', 'FE': 'green', 'Z': 'silver', 
               'C2': 'lightgrey', 'N2': 'lightblue'}

BOND_RADIUS = .12
BOND_COLOR = 'darkgrey'
BS_SCALE = .25
SPECULARITY = .7
SPEC_POWER = 80
CAMERA_SCALE = .5

FONTSIZE = 8
