# Initialization for the proteusPy package
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the MIT public license.

"""
.. include:: ../README.md
"""

__version__ = "0.81"

import sys
import os
import glob
import warnings
import copy

import time
import datetime
import math
import numpy

import Bio

from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.PDB import Select, Vector

from .ProteusPyWarning import ProteusPyWarning
from .ProteusGlobals import PDB_DIR, MODEL_DIR

from .DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException

from .turtle3D import Turtle3D
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN
from .Residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

from .Disulfide import Download_Disulfides, Extract_Disulfides
from .DisulfideList import load_disulfides_from_id
from .DisulfideLoader import DisulfideLoader, Load_PDB_SS

from .atoms import *
from .data import *
from .angle_annotation import *

from .utility import distance_squared, distance3d, get_jet_colormap, Check_chains, print_memory_used
from .utility import image_to_ascii_art, generate_vector_dataframe

from .DisulfideClasses import create_classes, create_quat_classes, plot_count_vs_classid, \
            plot_count_vs_class_df, get_half_quadrant, get_quadrant, get_sixth_quadrant, \
            filter_by_percentage, torsion_to_sixclass, plot_binary_to_sixclass_incidence, \
            enumerate_sixclass_fromlist
from .DisulfideClass_Constructor import DisulfideClass_Constructor

# end of file
