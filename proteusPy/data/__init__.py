# init for proteusPy data module
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.
# Cα Cβ Sγ Χ1 - Χ5 Χ

import os
import proteusPy

# global directory for PDB files and the correspondig .pkl models
abspath = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = f'{abspath}/'
SS_PICKLE_FILE = 'PDB_all_ss.pkl'
SS_DICT_PICKLE_FILE = 'PDB_all_ss_dict.pkl'
SS_TORSIONS_FILE = 'PDB_all_SS_torsions.csv'
PROBLEM_ID_FILE = 'PDB_all_SS_problems.csv'
SS_ID_FILE = 'ss_ids.txt'

# end of file
