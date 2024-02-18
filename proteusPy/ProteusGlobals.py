'''
This file contains global declarations for the *proteusPy* package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 2/18/24 -egs-
'''

import os

PDB_DIR = os.getenv('PDB')
PROTEUSPY_REPO_HOME = os.getenv("PROTEUSPY_REPO_HOME")
REPO_DATA_DIR = f'{PROTEUSPY_REPO_HOME}data/'

MODEL_DIR = f'{PDB_DIR}good/'

WINFRAME = 512 # single panel width
WINSIZE = (1024, 1024)
CAMERA_POS = ((0, 0, -10), (0,0,0), (0,1,0))

# global for initialization of dihedrals and energies
_FLOAT_INIT = -999.99
_INT_INIT = -1
_ANG_INIT = -180.0

PBAR_COLS = 105
# nominal macbook Pro screen resolution
DPI = 220

