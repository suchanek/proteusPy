
# global directories for PDB files and the correspondig .pkl models
PDB_DIR = './pdb/'
MODEL_DIR = f'{PDB_DIR}models/'
DATA_DIR = './data/'
SS_ID_FILE = f'{MODEL_DIR}ss_ids.txt'
SS_PICKLE_FILE = f'{MODEL_DIR}PDB_all_ss.pkl'
SS_DICT_PICKLE_FILE = f'{MODEL_DIR}PDB_all_ss_dict.pkl'
SS_TORSIONS_FILE = f'{MODEL_DIR}PDB_SS_torsions.csv'
PROBLEM_ID_FILE = f'{MODEL_DIR}PDB_SS_problems.csv'

# global for initialization of dihedrals and energies

ORIENT_BACKBONE = 2
ORIENT_SIDECHAIN = 1

# Cα Cβ Sγ Χ1 - Χ5 Χ

