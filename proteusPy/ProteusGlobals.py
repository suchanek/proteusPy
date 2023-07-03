'''
This file contains global declarations for the *proteusPy* package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 7/2/2023
'''
# Cα Cβ Sγ Χ1 - Χ5 Χ
#
# global directories for PDB files and the data and folders
# this is relative to the proteusPy package folder hierarchy!

import os

PDB_DIR = os.getenv('PDB')
PROTEUSPY_REPO_HOME = os.getenv("PROTEUSPY_REPO_HOME")
REPO_DATA_DIR = f'{PROTEUSPY_REPO_HOME}data/'

MODEL_DIR = f'{PDB_DIR}good/'
# global for initialization of dihedrals and energies

ORIENT_BACKBONE = 2
ORIENT_SIDECHAIN = 1

WINFRAME = 512 # single panel width
WINSIZE = (1024, 1024)
CAMERA_POS = ((0, 0, -10), (0,0,0), (0,1,0))

PBAR_COLS = 105
# nominal macbook Pro screen resolution
DPI = 220

