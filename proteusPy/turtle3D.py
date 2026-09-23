"""
3D turtle. The implementation lives in the ``turtlend`` package,
https://github.com/Flux-Frontiers/turtlend. This module re-exports it so that
``proteusPy.turtle3D`` keeps resolving. Full documentation:
https://flux-frontiers.github.io/turtlend/api/turtle3D/

Author: Eric G. Suchanek, PhD
"""

from turtlend.turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN, Turtle3D

__all__ = ["ORIENT_BACKBONE", "ORIENT_SIDECHAIN", "Turtle3D"]
