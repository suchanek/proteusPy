"""
Global declarations for the proteusPy package
"""

# init for proteusPy data module
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.

import os

HOME_DIR = os.path.expanduser("~")
PDB_BASE = os.getenv("PDB")

if PDB_BASE is None:
    print(f"The PDB environment variable is not set. Defaulting to {HOME_DIR}.")
    PDB_BASE = HOME_DIR


from .data import( 
    DATA_DIR, SS_CLASS_DEFINITIONS, 
    SS_CLASS_DICT_FILE, SS_CONSENSUS_FILE, SS_DICT_PICKLE_FILE,
    SS_DICT_PICKLE_FILE2, SS_ID_FILE, SS_PICKLE_FILE, SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_DICT_PICKLE_FILE, SS_SUBSET_DICT_PICKLE_FILE_IND, SS_SUBSET_PICKLE_FILE, SS_SUBSET_TORSIONS_FILE, 
    SS_TORSIONS_FILE, PROBLEM_ID_FILE, LOADER_FNAME_URL, LOADER_SUBSET_FNAME_URL, LOADER_ALL_URL, LOADER_SUBSET_URL, LOADER_FNAME,
    LOADER_SUBSET_FNAME, CLASSOBJ_FNAME)



