"""
Implementation of an N-dimensional 'Turtle' in Python.

Part of the program proteusPy, https://github.com/suchanek/proteusPy,
a Python package for the manipulation and analysis of macromolecules.

The TurtleND generalizes the classic 3D turtle graphics concept to arbitrary
dimensions. The turtle maintains a position and an orthonormal frame (a set of
N mutually orthogonal unit vectors) in N-dimensional space. Movement occurs
along the first basis vector (the "heading"), and rotations occur in planes
defined by pairs of basis vectors via Givens rotations.

This enables navigation through N-dimensional embedding spaces, providing
a geometric, frame-based alternative to gradient descent.

Author: Eric G. Suchanek, PhD
Based on the Turtle3D implementation.
"""

__pdoc__ = {"__all__": True}

import math

import numpy
import numpy as np

numpy.set_printoptions(suppress=True)


class TurtleND:
    """N-dimensional Turtle.

    The turtle maintains:
      - A position vector in R^n
      - An orthonormal frame of n basis vectors stored as rows of an (n x n) matrix

    By convention:
      - frame[0] = heading (direction of movement)
      - frame[1] = left
      - frame[2] = up  (when n >= 3)
      - frame[3..n-1] = higher-dimensional basis vectors

    Rotations are Givens rotations in the plane spanned by two basis vectors.
    The classic 3D operations (roll, yaw, turn, pitch) are special cases.
    """

    def __init__(self, ndim: int = 3, name: str = "ND_Turtle"):
        """
        Initialize an N-dimensional turtle at the origin with the standard
        orthonormal frame.

        :param ndim: Number of dimensions (must be >= 2)
        :param name: Turtle's name
        """
        if ndim < 2:
            raise ValueError("TurtleND requires at least 2 dimensions")

        self._ndim = ndim
        self._name = name
        self._position = np.zeros(ndim, dtype="d")
        self._frame = np.eye(ndim, dtype="d")  # rows are basis vectors
        self._pen = 1  # up
        self._recording = False
        self._tape = []

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    @property
    def name(self) -> str:
        """Turtle's name."""
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def position(self) -> np.ndarray:
        """The turtle's position as a copy."""
        return self._position.copy()

    @position.setter
    def position(self, pos) -> None:
        pos = np.asarray(pos, dtype="d")
        if pos.shape != (self._ndim,):
            raise ValueError(f"Position must have shape ({self._ndim},)")
        self._position = pos

    @property
    def frame(self) -> np.ndarray:
        """The full orthonormal frame as a copy. Rows are basis vectors."""
        return self._frame.copy()

    @property
    def heading(self) -> np.ndarray:
        """Basis vector 0: the heading direction."""
        return self._frame[0].copy()

    @property
    def left(self) -> np.ndarray:
        """Basis vector 1: the left direction."""
        return self._frame[1].copy()

    @property
    def up(self) -> np.ndarray:
        """Basis vector 2: the up direction (requires ndim >= 3)."""
        if self._ndim < 3:
            raise ValueError("'up' requires at least 3 dimensions")
        return self._frame[2].copy()

    def basis(self, i: int) -> np.ndarray:
        """
        Return a copy of the i-th basis vector.

        :param i: Basis vector index (0 = heading, 1 = left, 2 = up, ...)
        """
        if i < 0 or i >= self._ndim:
            raise IndexError(f"Basis index {i} out of range for {self._ndim}D turtle")
        return self._frame[i].copy()

    def reset(self) -> None:
        """Reset the turtle to the origin with the identity frame."""
        self._position = np.zeros(self._ndim, dtype="d")
        self._frame = np.eye(self._ndim, dtype="d")

    def copy_coords(self, source: "TurtleND") -> None:
        """
        Copy position and frame from another TurtleND.

        :param source: Source turtle (must have the same ndim)
        """
        if source._ndim != self._ndim:
            raise ValueError(
                f"Cannot copy from {source._ndim}D turtle to {self._ndim}D turtle"
            )
        self._position = source._position.copy()
        self._frame = source._frame.copy()

    # ---------------------------------------------------------------
    # Movement
    # ---------------------------------------------------------------

    def move(self, distance: float) -> None:
        """
        Move the turtle along the heading vector by the given distance.

        :param distance: Distance to move
        """
        self._position = self._position + self._frame[0] * distance
        if self._recording:
            self._tape.append(self._position.copy())

    # ---------------------------------------------------------------
    # Rotation primitives
    # ---------------------------------------------------------------

    def _rotate(self, angle_rad: float, i: int, j: int) -> None:
        """
        Apply a Givens rotation in the plane of basis[i] and basis[j].

        Rotates basis[i] toward basis[j] by angle_rad radians.
        This is the fundamental rotation primitive for N dimensions.

        :param angle_rad: Rotation angle in radians
        :param i: First basis vector index
        :param j: Second basis vector index
        """
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)

        bi = self._frame[i].copy()
        bj = self._frame[j].copy()

        self._frame[i] = c * bi + s * bj
        self._frame[j] = c * bj - s * bi

        # re-normalize to prevent drift
        self._frame[i] /= np.linalg.norm(self._frame[i])
        self._frame[j] /= np.linalg.norm(self._frame[j])

    def rotate(self, angle: float, i: int, j: int) -> None:
        """
        Rotate in the plane of basis[i] and basis[j] by angle degrees.

        This is the general N-dimensional rotation. All named rotations
        (roll, turn, pitch, yaw) are special cases.

        :param angle: Rotation angle in degrees
        :param i: First basis vector index
        :param j: Second basis vector index
        """
        if i < 0 or i >= self._ndim or j < 0 or j >= self._ndim:
            raise IndexError(f"Basis indices ({i}, {j}) out of range for {self._ndim}D")
        if i == j:
            raise ValueError("Cannot rotate in a degenerate plane (i == j)")
        self._rotate(math.radians(angle), i, j)

    # ---------------------------------------------------------------
    # Classic 3D-compatible named rotations
    # ---------------------------------------------------------------

    def roll(self, angle: float) -> None:
        """
        Roll: rotate in the left-up plane (basis[1] toward basis[2]).

        In 3D this rotates about the heading axis.

        :param angle: Roll angle in degrees
        """
        if self._ndim < 3:
            raise ValueError("roll requires at least 3 dimensions")
        # Matches Turtle3D: up rotates toward -left, left rotates toward up
        # i.e., basis[2] toward -basis[1] = basis[1] toward basis[2] with negated angle
        # Let's match the exact Turtle3D convention:
        #   up_new  = cos(a)*up  - sin(a)*left
        #   left_new = cos(a)*left + sin(a)*up
        # That's rotating basis[1] toward basis[2] by +angle
        self._rotate(math.radians(angle), 1, 2)

    def turn(self, angle: float) -> None:
        """
        Turn: rotate in the heading-left plane (basis[0] toward basis[1]).

        In 3D this rotates about the up axis.

        :param angle: Turn angle in degrees
        """
        self._rotate(math.radians(angle), 0, 1)

    def pitch(self, angle: float) -> None:
        """
        Pitch: rotate in the heading-up plane (basis[0] toward basis[2]).

        In 3D this rotates about the left axis. Negative angle pitches
        the heading downward (toward -up), matching Turtle3D convention.

        :param angle: Pitch angle in degrees
        """
        if self._ndim < 3:
            raise ValueError("pitch requires at least 3 dimensions")
        # Turtle3D convention:
        #   heading_new = cos(a)*heading - sin(a)*up
        #   up_new      = cos(a)*up      + sin(a)*heading
        # That's basis[0] toward -basis[2], i.e., rotating by -angle in (0,2)
        self._rotate(math.radians(-angle), 0, 2)

    def yaw(self, angle: float) -> None:
        """
        Yaw: rotate in the heading-left plane by (180 - angle) degrees.

        This matches the Turtle3D convention used for molecular building.

        :param angle: Yaw angle in degrees
        """
        self._rotate(math.radians(180.0 - angle), 0, 1)

    # ---------------------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------------------

    def to_local(self, global_vec) -> np.ndarray:
        """
        Transform a global-space vector to turtle-local coordinates.

        Projects (global_vec - position) onto each basis vector.

        :param global_vec: Point in global coordinates (array-like of length ndim)
        :return: Local coordinates as ndarray of shape (ndim,)
        """
        global_vec = np.asarray(global_vec, dtype="d")
        delta = global_vec - self._position
        # Each component is the dot product with the corresponding basis vector
        return self._frame @ delta

    def to_global(self, local_vec) -> np.ndarray:
        """
        Transform turtle-local coordinates to global-space coordinates.

        global = position + sum_i(local[i] * basis[i])

        :param local_vec: Point in local coordinates (array-like of length ndim)
        :return: Global coordinates as ndarray of shape (ndim,)
        """
        local_vec = np.asarray(local_vec, dtype="d")
        # frame.T @ local_vec gives sum of local[i] * basis[i] (columns of frame.T)
        return self._position + self._frame.T @ local_vec

    # ---------------------------------------------------------------
    # Frame maintenance
    # ---------------------------------------------------------------

    def orthonormalize(self) -> None:
        """
        Re-orthonormalize the frame using modified Gram-Schmidt.

        Call this periodically if accumulated floating-point drift is a concern.
        """
        q, _ = np.linalg.qr(self._frame.T)
        # Preserve orientation: ensure each new basis vector is in the same
        # half-space as the original
        for i in range(self._ndim):
            if np.dot(q[:, i], self._frame[i]) < 0:
                q[:, i] = -q[:, i]
        self._frame = q.T

    def orient(self, position, heading, left) -> None:
        """
        Orient the turtle at position, with heading and left directions.
        Remaining basis vectors are constructed via Gram-Schmidt.

        :param position: New position (array-like of length ndim)
        :param heading: Heading direction (array-like of length ndim)
        :param left: Left direction (array-like of length ndim)
        """
        position = np.asarray(position, dtype="d")
        heading = np.asarray(heading, dtype="d")
        left = np.asarray(left, dtype="d")

        self._position = position.copy()

        # Build heading
        h = heading - position
        h = h / np.linalg.norm(h)
        self._frame[0] = h

        # Build left, orthogonal to heading
        left_vec = left - position
        left_vec = left_vec - np.dot(left_vec, h) * h
        left_vec = left_vec / np.linalg.norm(left_vec)
        self._frame[1] = left_vec

        # Build remaining basis vectors via Gram-Schmidt from random vectors
        # seeded deterministically
        for k in range(2, self._ndim):
            # Start with the k-th standard basis vector
            v = np.zeros(self._ndim, dtype="d")
            v[k] = 1.0
            # Subtract projections onto all previously determined basis vectors
            for prev in range(k):
                v = v - np.dot(v, self._frame[prev]) * self._frame[prev]
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                # Fallback: try other standard basis vectors
                for alt in range(self._ndim):
                    v = np.zeros(self._ndim, dtype="d")
                    v[alt] = 1.0
                    for prev in range(k):
                        v = v - np.dot(v, self._frame[prev]) * self._frame[prev]
                    norm = np.linalg.norm(v)
                    if norm > 1e-12:
                        break
            self._frame[k] = v / norm

    # ---------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------

    @property
    def pen(self) -> str:
        return "up" if self._pen == 1 else "down"

    @pen.setter
    def pen(self, value: str):
        self._pen = 1 if value == "up" else -1

    @property
    def recording(self) -> bool:
        return self._recording

    @recording.setter
    def recording(self, value: bool):
        self._recording = value

    @property
    def tape(self) -> list:
        """Return a copy of the recorded positions."""
        return list(self._tape)

    def reset_tape(self) -> None:
        self._recording = False
        self._tape = []

    def __repr__(self):
        basis_strs = "\n".join(
            f"  basis[{i}]: {self._frame[i]}" for i in range(self._ndim)
        )
        return (
            f"<TurtleND: {self._name} ({self._ndim}D)\n"
            f" Position: {self._position}\n"
            f" Frame:\n{basis_strs}\n"
            f" Pen: {self.pen}, Recording: {self._recording}>"
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
