"""
N-dimensional turtle. The implementation lives in the ``turtlend`` package,
https://github.com/Flux-Frontiers/turtlend. This module re-exports it so that
``proteusPy.turtleND`` keeps resolving. Full documentation:
https://flux-frontiers.github.io/turtlend/api/turtleND/

Author: Eric G. Suchanek, PhD
"""

from turtlend.turtleND import TurtleND

__all__ = ["TurtleND"]
