# Initialization for the proteusPy package
# Copyright (c) 2025 Eric G. Suchanek, PhD., all rights reserved
# Subject to the BSD public license.
# Last updated: 2025-03-27 19:48:55 -egs-

# pylint: disable=C0413
# pylint: disable=C0103

"""
``proteusPy`` is a Python package specializing in the modeling and analysis
of proteins of known structure with an emphasis on disulfide bonds. This package
reprises the molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023),
a structure-based program developed as part of Eric Suchanek's graduate thesis,
which was originally written in Common Lisp, ported to C++ and ultimately to Python.

The package utilizes several base classes to create and analyze disulfide bonds:
- Turtle3D: to build disulfides through the manipulation of local coordinate systems.
- DisulfideBase: basic characteristics of individual disulfide bonds, classes ``Disulfide`` and ``DisulfideList``
- DisulfideClassManager: to manage ``Disulfide`` classes
- DisulfideClassGenerator: to generate ``Disulfide`` structure ensembles from disulfide classes
- DisulfideStats: to calculate statistics on disulfide bonds
- DisulfideVisualization: to visualize disulfide bonds
- DisulfideLoader: to load disulfide bonds from the master list of Disulfides and create the
 various data structures needed to build the structural classes and physical properties of
 the disulfide bonds.


This implementation of ``proteusPy`` focuses on the ``Disulfide`` class. This class
implements methods to analyze the protein structure stabilizing element known as a
*Disulfide bond*. Its underlying methods  have been utilized to construct a database of over 36,000
high-quality disulfide bonds from the RCSB protein data bank (https://www.rcsb.org) for structural analysis.
"""

__pdoc__ = {
    "version": None,
    "__all__": False,
    "_version": False,
}

import logging

# Set the default (global) logger level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)

# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


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
    SPEC_POWER,
    SPECULARITY,
)
from .DisulfideBase import Disulfide, DisulfideList
from .DisulfideClasses import (
    angle_within_range,
    filter_by_percentage,
    get_quadrant,
    is_between,
)
from .DisulfideClassGenerator import DisulfideClassGenerator
from .DisulfideClassManager import DisulfideClassManager
from .DisulfideExceptions import (
    DisulfideConstructionException,
    DisulfideConstructionWarning,
    DisulfideException,
    DisulfideIOException,
    DisulfideParseWarning,
)
from .DisulfideIO import (
    Initialize_Disulfide_From_Coords,
    extract_disulfide,
    load_disulfides_from_id,
)
from .DisulfideLoader import Bootstrap_PDB_SS, DisulfideLoader, Load_PDB_SS
from .DisulfideStats import DisulfideStats
from .DisulfideVisualization import DisulfideVisualization
from .logger_config import (
    DEFAULT_LOG_LEVEL,
    configure_master_logger,
    create_logger,
    disable_stream_handlers_for_namespace,
    list_all_loggers,
    list_handlers,
    set_logger_level,
    set_logger_level_for_module,
    set_logging_level_for_all_handlers,
    toggle_stream_handler,
)
from .Plotting import plot_class_chart
from .ProteusGlobals import (
    _ANG_INIT,
    _FLOAT_INIT,
    _INT_INIT,
    BINARY_CLASS_METRICS_CSV_FILE,
    BINARY_CLASS_METRICS_FILE,
    CA_CUTOFF,
    CAMERA_POS,
    FONTSIZE,
    LOADER_ALL_MASTER_URL,
    LOADER_FNAME,
    LOADER_FNAME_URL,
    LOADER_SUBSET_FNAME,
    LOADER_SUBSET_FNAME_URL,
    LOADER_SUBSET_MASTER_URL,
    MODEL_DIR,
    OCTANT_CLASS_METRICS_CSV_FILE,
    OCTANT_CLASS_METRICS_FILE,
    PDB_DIR,
    PROBLEM_ID_FILE,
    SG_CUTOFF,
    SS_CLASS_DEFINITIONS,
    SS_CLASS_DICT_FILE,
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
    SS_ID_FILE,
    SS_LIST_URL,
    SS_MASTER_PICKLE_FILE,
    SS_PICKLE_FILE,
    SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_PICKLE_FILE,
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
    Extract_Disulfides,
    Extract_Disulfides_From_List,
    display_ss_pymol,
    distance_squared,
    extract_firstchain_ss,
    generate_vector_dataframe,
    get_jet_colormap,
    get_memory_usage,
    get_object_size_mb,
    get_theme,
    grid_dimensions,
    image_to_ascii_art,
    load_list_from_file,
    print_memory_used,
    prune_extra_ss,
    remove_duplicate_ss,
    retrieve_git_lfs_files,
    save_list_to_file,
    set_plotly_theme,
    set_pyvista_theme,
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

set_plotly_theme(theme="auto")
set_pyvista_theme(theme="auto")

# Create a logger for the package itself. __name__ is the package name, proteusPy

_logger = create_logger(__name__, log_level=DEFAULT_LOG_LEVEL)
configure_master_logger("proteusPy.log", disabled=True)


def describe():
    """
    Describe the proteusPy package.
    """
    set_logger_level_for_module("proteusPy", logging.INFO)
    _logger.info(
        f"ProteusPy {__version__} initialized. Plotly theme set to: {set_plotly_theme(theme='auto')}. PyVista theme set to: {set_pyvista_theme(theme='auto')}. Logging level setting to default: {DEFAULT_LOG_LEVEL}"
    )
    _logger.setLevel(DEFAULT_LOG_LEVEL)
    return


# end of file
