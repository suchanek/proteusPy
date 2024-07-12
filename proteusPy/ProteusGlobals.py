"""
This file contains global declarations for the *proteusPy* package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 7/12/24 -egs-
"""

import math
import os
from pathlib import Path

_this_dir = Path(__file__).parent

HOME_DIR = os.path.expanduser("~")
PDB_DIR = os.getenv("PDB")

if PDB_DIR is None:
    print(f"The PDB environment variable is not set. Defaulting to {HOME_DIR}.")
    PDB_DIR = HOME_DIR


DATA_DIR = os.path.join(_this_dir, "data", "")
REPO_DATA_DIR = os.path.join(_this_dir, "data", "")
MODEL_DIR = os.path.join(PDB_DIR, "good", "")

WINFRAME = 512  # single panel width
WINSIZE = (1024, 1024)
CAMERA_POS = ((0, 0, -10), (0, 0, 0), (0, 1, 0))

# global for initialization of dihedrals and energies
_FLOAT_INIT = math.nan
_INT_INIT = -1
_ANG_INIT = -180.0

PBAR_COLS = 105
# nominal macbook Pro screen resolution
DPI = 220

# global names for the disulfide .pkl files

SS_PICKLE_FILE = "PDB_all_ss.pkl"
SS_DICT_PICKLE_FILE = "PDB_all_ss_dict.pkl"
SS_DICT_PICKLE_FILE2 = "PDB_all_ss_dict_ind.pkl"
SS_TORSIONS_FILE = "PDB_all_ss_torsions.csv"
PROBLEM_ID_FILE = "PDB_all_SS_problems.csv"
SS_ID_FILE = "ss_ids.txt"

SS_SUBSET_PICKLE_FILE = "PDB_subset_ss.pkl"
SS_SUBSET_DICT_PICKLE_FILE = "PDB_subset_ss_dict.pkl"
SS_SUBSET_DICT_PICKLE_FILE_IND = "PDB_subset_ss_dict_ind.pkl"
SS_SUBSET_TORSIONS_FILE = "PDB_subset_torsions.csv"
SS_PROBLEM_SUBSET_ID_FILE = "PDB_subset_problems.csv"

# contains the dihedral classes and their members
SS_CLASS_DICT_FILE = "PDB_ss_classes_dict.pkl"
SS_CONSENSUS_FILE = "SS_consensus_class_sext.pkl"

LOADER_FNAME = "PDB_SS_ALL_LOADER.pkl"
LOADER_SUBSET_FNAME = "PDB_SS_SUBSET_LOADER.pkl"

LOADER_FNAME_URL = "https://raw.githubusercontent.com/suchanek/proteusPy/master/proteusPy/data/PDB_SS_ALL_LOADER.pkl"
LOADER_SUBSET_FNAME_URL = "https://raw.githubusercontent.com/suchanek/proteusPy/blob/master/proteusPy/data/PDB_SS_SUBSET_LOADER.pkl"


LOADER_ALL_URL = "https://drive.google.com/uc?id=1igF-sppLPaNsBaUS7nkb13vtOGZZmsFp"
LOADER_SUBSET_URL = "https://drive.google.com/uc?id=1puy9pxrClFks0KN9q5PPV_ONKvL-hg33"

SS_CLASS_DEFINITIONS = """
Idx,chi1_s,chi2_s,chi3_s,chi4_s,chi5_s,class_id,SS_Classname,FXN
0,-1,-1,-1,-1,-1,00000,-LHSpiral,UNK
1,-1,-1,-1,-1,1,00002,00002,UNK
2,-1,-1,-1,1,-1,00020,-LHHook,UNK
3,-1,-1,-1,1,1,00022,00022,UNK
4,-1,-1,1,-1,-1,00200,-RHStaple,Allosteric
5,-1,-1,1,-1,1,00202,00202,UNK
6,-1,-1,1,1,-1,00220,00220,UNK
7,-1,-1,1,1,1,00222,00222,UNK
8,-1,1,-1,-1,-1,02000,02000,UNK
9,-1,1,-1,-1,1,02002,02002,UNK
10,-1,1,-1,1,-1,02020,-LHStaple,UNK
11,-1,1,-1,1,1,02022,02022,UNK
12,-1,1,1,-1,-1,02200,-RHHook,UNK
13,-1,1,1,-1,1,02202,02202,UNK
14,-1,1,1,1,-1,02220,-RHSpiral,UNK
15,-1,1,1,1,1,02222,02222,UNK
16,1,-1,-1,-1,-1,20000,±LHSpiral,UNK
17,1,-1,-1,-1,1,20002,+LHSpiral,UNK
18,1,-1,-1,1,-1,20020,±LHHook,UNK
19,1,-1,-1,1,1,20022,+LHHook,UNK
20,1,-1,1,-1,-1,20200,±RHStaple,UNK
21,1,-1,1,-1,1,20202,+RHStaple,UNK
22,1,-1,1,1,-1,20220,±RHHook,Catalytic
23,1,-1,1,1,1,20222,20222,UNK
24,1,1,-1,-1,-1,22000,-/+LHHook,UNK
25,1,1,-1,-1,1,22002,22002,UNK
26,1,1,-1,1,-1,22020,±LHStaple,UNK
27,1,1,-1,1,1,22022,+LHStaple,UNK
28,1,1,1,-1,-1,22200,-/+RHHook,UNK
29,1,1,1,-1,1,22202,+RHHook,UNK
30,1,1,1,1,-1,22220,±RHSpiral,UNK
31,1,1,1,1,1,22222,+RHSpiral,UNK
"""

CLASSOBJ_FNAME = "PDB_CLASS_OBJ.pkl"
