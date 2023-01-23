
'''
A package for the modeling and analysis of proteins of known structure.
With an emphasis on Disulfide Bonds. This package reprises my molecular 
modeling and analysis program [Proteus](https://doi.org/10.1021/bi00368a023), 
The package relies on the ``Turtle3D`` class, which implements a 
three-dimensional 'Turtle'. The turtle implements ``Move``, ``Roll``, 
``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space. 
The ```DisulfideBond``` class implements the protein structure stabilizing element 
known as a Disulfide Bond. This class and its underlying methods are being used to 
perform an analysis of proteins contained in the RCSB protein data bank.

Eric G. Suchanek, PhD.
suchanek@mac.com

'''
# Initialization for the proteusPy package
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the MIT public license.

__version__ = "0.9dev"

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

from .Disulfide import Download_Disulfides, Extract_Disulfides, Check_chains

from .DisulfideLoader import DisulfideLoader
from .atoms import *
from .utility import distance_squared, distance3d, cmap_vector

# end of file
