#
# global directories for PDB files and the correspondig .pkl models
# this is relative to the proteusPy package folder hierarchy!
#
import os

PDB_DIR = os.getenv('PDB')
MODEL_DIR = f'{PDB_DIR}/good/'
# global for initialization of dihedrals and energies

ORIENT_BACKBONE = 2
ORIENT_SIDECHAIN = 1
# Cα Cβ Sγ Χ1 - Χ5 Χ


