# init for proteusPy data module
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.

import os

# absolute location for the disulfide .pkl files
_abspath = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = f'{_abspath}/'
SS_PICKLE_FILE = 'PDB_all_ss.pkl'
SS_DICT_PICKLE_FILE = 'PDB_all_ss_dict.pkl'
SS_DICT_PICKLE_FILE2 = 'PDB_all_ss_dict_ind.pkl'
SS_TORSIONS_FILE = 'PDB_all_SS_torsions.csv'
PROBLEM_ID_FILE = 'PDB_all_SS_problems.csv'
SS_ID_FILE = 'ss_ids.txt'

SS_SUBSET_PICKLE_FILE = 'PDB_subset_ss.pkl'
SS_SUBSET_DICT_PICKLE_FILE = 'PDB_subset_ss_dict.pkl'
SS_SUBSET_DICT_PICKLE_FILE_IND = 'PDB_subset_ss_dict_ind.pkl'
SS_SUBSET_TORSIONS_FILE = 'PDB_subset_torsions.csv'
SS_PROBLEM_SUBSET_ID_FILE = 'PDB_subset_problems.csv'

# end of file
