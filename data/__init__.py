'''
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
The filenames defined herein are used by the [DisulfideLoader](proteusPy.DisulfideLoader)
class to build the disulfide database.

Author: Eric G. Suchanek, PhD
Last revision: 2/14/2023
'''

# init for proteusPy data module
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.

import os
import proteusPy

# absolute location for the disulfide .pkl files
_abspath = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = f'{_abspath}/'
SS_PICKLE_FILE = 'PDB_all_ss.pkl'
SS_DICT_PICKLE_FILE = 'PDB_all_ss_dict.pkl'
SS_TORSIONS_FILE = 'PDB_all_SS_torsions.csv'
PROBLEM_ID_FILE = 'PDB_all_SS_problems.csv'
SS_ID_FILE = 'ss_ids.txt'

# end of file
