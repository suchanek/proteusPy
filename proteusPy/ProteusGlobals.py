"""
This file contains global declarations for the *proteusPy* package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 3/12/24 -egs-
"""

import math
import os
from pathlib import Path

_this_dir = Path(__file__).parent

PDB_DIR = os.getenv("PDB")
if PDB_DIR is None:
    PDB_DIR = "."

REPO_DATA_DIR = _this_dir / "data"

MODEL_DIR = os.path.join(PDB_DIR, "good/")

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
