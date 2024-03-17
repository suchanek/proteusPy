# Initialization for the proteusPy package
# Copyright (c) 2024 Eric G. Suchanek, PhD., all rights reserved
# Subject to the BSD public license.

"""
.. include:: ../README.md
"""

from pathlib import Path

__pdoc__ = {"__all__": False}

"""
_version_file = Path(__file__).parent.parent / "VERSION"
if _version_file.is_file():
    with open(_version_file) as f:
        __version__ = f.read().strip()

__version__ = "0.93.0"
"""

import copy
import datetime
import glob
import math
import os
import sys
import time
import warnings

import Bio
import matplotlib
import numpy
import plotly
from Bio.PDB import Select, Vector
from Bio.PDB.vectors import calc_angle, calc_dihedral

from .angle_annotation import *
from .atoms import *
from .data import *
from .Disulfide import (
    Disulfide_Energy_Function,
    Download_Disulfides,
    Extract_Disulfides,
    check_header_from_file,
    check_header_from_id,
)
from .DisulfideClass_Constructor import DisulfideClass_Constructor
from .DisulfideClasses import (
    create_classes,
    create_quat_classes,
    enumerate_sixclass_fromlist,
    filter_by_percentage,
    get_half_quadrant,
    get_quadrant,
    get_sixth_quadrant,
    plot_binary_to_sixclass_incidence,
    plot_count_vs_class_df,
    plot_count_vs_classid,
    torsion_to_sixclass,
)
from .DisulfideExceptions import (
    DisulfideConstructionException,
    DisulfideConstructionWarning,
    DisulfideIOException,
)
from .DisulfideList import load_disulfides_from_id
from .DisulfideLoader import Download_PDB_SS, Download_PDB_SS_GitHub, Load_PDB_SS
from .ProteusGlobals import (
    _ANG_INIT,
    _FLOAT_INIT,
    _INT_INIT,
    CAMERA_POS,
    MODEL_DIR,
    PDB_DIR,
    WINFRAME,
    WINSIZE,
)
from .ProteusPyWarning import ProteusPyWarning
from .Residue import (
    build_residue,
    get_backbone_from_chain,
    to_alpha,
    to_carbonyl,
    to_nitrogen,
    to_oxygen,
)
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN, Turtle3D
from .utility import (
    Check_chains,
    distance3d,
    distance_squared,
    generate_vector_dataframe,
    get_jet_colormap,
    image_to_ascii_art,
    print_memory_used,
    retrieve_git_lfs_files,
)
from .version import __version__

# end of file
