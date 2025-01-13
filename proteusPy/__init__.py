# Initialization for the proteusPy package
# Copyright (c) 2024 Eric G. Suchanek, PhD., all rights reserved
# Subject to the BSD public license.
# Last updated: 2025-01-07 18:38:30 -egs-

# pylint: disable=C0413

"""
.. include:: ../README.md
"""

__pdoc__ = {
    "version": None,
    "__all__": False,
    "_version": False,
}

import logging

DEFAULT_LOG_LEVEL = logging.WARNING

# Set the default (global) logger level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)

# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Create a logger for the package itself. __name__ is the package name, proteusPy
_logger = logging.getLogger(__name__)

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
    Initialize_Disulfide_From_Coords,
    disulfide_energy_function,
    minimize_ss_energy,
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
from .DisulfideList import (
    DisulfideList,
    calculate_torsion_statistics,
    extract_disulfide,
    load_disulfides_from_id,
)
from .DisulfideLoader import Bootstrap_PDB_SS, DisulfideLoader, Load_PDB_SS
from .logger_config import (
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
from .ProteusGlobals import (
    _ANG_INIT,
    _FLOAT_INIT,
    _INT_INIT,
    CA_CUTOFF,
    CAMERA_POS,
    LOADER_ALL_MASTER_URL,
    LOADER_FNAME,
    LOADER_FNAME_URL,
    LOADER_SUBSET_FNAME,
    LOADER_SUBSET_FNAME_URL,
    LOADER_SUBSET_MASTER_URL,
    MODEL_DIR,
    PDB_DIR,
    PROBLEM_ID_FILE,
    SG_CUTOFF,
    SS_CLASS_DEFINITIONS,
    SS_CLASS_DICT_FILE,
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
    SS_CONSENSUS_SEXT_FILE,
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
    calculate_percentile_cutoff,
    calculate_std_cutoff,
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

_logger.info("ProteusPy %s initialized ", __version__)
_logger.setLevel(DEFAULT_LOG_LEVEL)

# end of file
