"""
3D vector class and angle, dihedral and distance helpers. The implementation
lives in the ``turtlend`` package, https://github.com/Flux-Frontiers/turtlend.
This module re-exports it so that ``proteusPy.vector3D`` keeps resolving.

Author: Eric G. Suchanek, PhD
"""

import logging

from turtlend.vector3D import (
    Vector3D,
    calc_angle,
    calc_dihedral,
    calculate_bond_angle,
    distance3d,
    rms_difference,
)

# proteusPy's own copy of this module logged at ERROR, so zero-vector warnings
# stayed quiet. Keep that: turtlend's logger has no level of its own.
logging.getLogger("turtlend.vector3D").setLevel(logging.ERROR)

__all__ = [
    "Vector3D",
    "calc_angle",
    "calc_dihedral",
    "calculate_bond_angle",
    "distance3d",
    "rms_difference",
]
