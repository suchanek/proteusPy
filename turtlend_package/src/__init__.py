# TurtleND - N-Dimensional Turtle Graphics & ManifoldModel
#
# Core modules:
#   turtle3D        - 3D turtle with coordinate frame (heading/left/up)
#   turtleND        - N-dimensional turtle via Givens rotations
#   manifold_walker - Manifold-aware navigation using local PCA
#   manifold_model  - ManifoldModel classifier ("the manifold IS the model")

from .manifold_model import ManifoldModel
from .manifold_walker import ManifoldWalker
from .turtleND import TurtleND
