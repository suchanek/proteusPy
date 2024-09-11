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
    "_version": False,
}

import logging

# Set the default logger level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)


# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np

from ._version import __version__
from .angle_annotation import AngleAnnotation, plot_angle
from .atoms import (
    ATOM_COLORS,
    ATOM_RADII_COVALENT,
    ATOM_RADII_CPK,
    BOND_COLOR,
    BOND_RADIUS,
    BS_SCALE,
    CAMERA_SCALE,
    FONTSIZE,
    SPEC_POWER,
    SPECULARITY,
)
from .Disulfide import (
    Disulfide,
    Disulfide_Energy_Function,
    Initialize_Disulfide_From_Coords,
    Minimize,
)
from .DisulfideClass_Constructor import DisulfideClass_Constructor
from .DisulfideClasses import (
    angle_within_range,
    create_classes,
    filter_by_percentage,
    get_angle_class,
    get_quadrant,
    get_section,
    get_ss_id,
    is_between,
    plot_class_chart,
    torsion_to_class_string,
    torsion_to_eightclass,
    torsion_to_sixclass,
)
from .DisulfideExceptions import (
    DisulfideConstructionException,
    DisulfideConstructionWarning,
    DisulfideException,
    DisulfideIOException,
    DisulfideParseWarning,
)
from .DisulfideList import DisulfideList, load_disulfides_from_id
from .DisulfideLoader import (
    Bootstrap_PDB_SS,
    DisulfideLoader,
    Download_PDB_SS,
    Download_PDB_SS_GitHub,
    Load_PDB_SS,
)
from .logger_config import get_logger, set_logger_level
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
from .ssparser import (
    check_file,
    extract_and_write_ssbonds_and_atoms,
    extract_ssbonds_and_atoms,
    get_atom_coordinates,
    get_phipsi_atoms_coordinates,
    get_residue_atoms_coordinates,
    print_disulfide_bond_info_dict,
)
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN, Turtle3D
from .utility import (
    Download_Disulfides,
    Extract_Disulfide,
    Extract_Disulfides,
    Extract_Disulfides_From_List,
    calculate_percentile_cutoff,
    calculate_std_cutoff,
    create_deviation_dataframe,
    display_ss_pymol,
    distance_squared,
    extract_firstchain_ss,
    generate_vector_dataframe,
    get_jet_colormap,
    get_macos_theme,
    get_memory_usage,
    get_object_size_mb,
    grid_dimensions,
    image_to_ascii_art,
    load_list_from_file,
    plot_class_chart,
    print_memory_used,
    prune_extra_ss,
    remove_duplicate_ss,
    retrieve_git_lfs_files,
    save_list_to_file,
    set_logger_level_for_module,
    sort_by_column,
)
from .vector3D import (
    Vector3D,
    calc_angle,
    calc_dihedral,
    calculate_bond_angle,
    distance3d,
    rms_difference,
)

# print(f"ProteusPy {__version__}")

# end of file
