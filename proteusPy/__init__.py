# Initialization for the proteusPy package
# Copyright (c) 2024 Eric G. Suchanek, PhD., all rights reserved
# Subject to the BSD public license.

"""
.. include:: ../README.md
"""

from pathlib import Path

__pdoc__ = {
    "version": None,
    "__all__": False,
}

import copy
import datetime
import glob
import math
import os
import pickle
import subprocess
import sys
import time
import warnings

# import Bio
import matplotlib.pyplot as plt
import numpy
import plotly
from Bio.PDB import Vector
from Bio.PDB.vectors import calc_dihedral

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from Bio.PDB import Select, Vector
from Bio.PDB.vectors import calc_angle, calc_dihedral

from .angle_annotation import *
from .atoms import *
from .data import *
from .Disulfide import Disulfide, Disulfide_Energy_Function, Minimize
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
from .DisulfideList import DisulfideList, load_disulfides_from_id
from .DisulfideLoader import (
    DisulfideLoader,
    Download_PDB_SS,
    Download_PDB_SS_GitHub,
    Load_PDB_SS,
)
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
    Download_Disulfides,
    Extract_Disulfide,
    Extract_Disulfides,
    check_header_from_file,
    check_header_from_id,
    display_ss_pymol,
    distance3d,
    distance_squared,
    extract_firstchain_ss,
    generate_vector_dataframe,
    get_jet_colormap,
    get_memory_usage,
    grid_dimensions,
    image_to_ascii_art,
    parse_ssbond_header_rec,
    plot_class_chart,
    print_memory_used,
    prune_extra_ss,
    remove_duplicate_ss,
    retrieve_git_lfs_files,
    sort_by_column,
)
from .version import __version__

# end of file
