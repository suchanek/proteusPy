# Initialization for the proteusPy package
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the MIT public license.

"""
.. include:: ../README.md
"""

__version__ = "0.51dev"

import sys
import os
import glob
import warnings
import copy

import time
import datetime
import math
import numpy

from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.PDB import Select, Vector

from .proteusPyWarning import ProteusPyWarning
from .ProteusGlobals import PDB_DIR, MODEL_DIR

from .DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException

from .turtle3D import Turtle3D
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN
from .residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

from .Disulfide import Download_Disulfides, Extract_Disulfides
from .DisulfideLoader import DisulfideLoader, Load_PDB_SS
from .DisulfideList import load_disulfides_from_id

from .atoms import *
from .data import *
from .utility import distance_squared, distance3d, cmap_vector, Check_chains

# end of file
