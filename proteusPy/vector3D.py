"""
This module is part of the proteusPy package and provides utility functions
for calculating geometric properties such as angles and dihedral angles
between vectors in 3D space. The vectors are represented using the Vector3D
class, which supports various vector operations.

Functions:
    calc_angle(v1, v2, v3):
        Calculate the angle between three vectors representing three connected points.

    calc_dihedral(v1, v2, v3, v4):
        Calculate the dihedral angle between four vectors representing four connected points.
"""

import math

import numpy as np

from proteusPy.logger_config import get_logger

_logger = get_logger(__name__)  ## vector3d


class Vector3D:
    """Class representing a 3D vector."""

    def __init__(self, x, y=None, z=None):
        """Initialize the vector with either a list/tuple/array of 3 elements or three separate values."""
        if y is None and z is None:
            if len(x) != 3:
                raise ValueError(
                    "Vector3D: Input must be a list/tuple/array of 3 elements"
                )
            self._coords = np.array(x, dtype=float)
        else:
            self._coords = np.array((x, y, z), dtype=float)

    def __repr__(self):
        """Return a string representation of the vector."""
        x, y, z = self._coords
        return f"<Vector3D {x:.2f}, {y:.2f}, {z:.2f}>"

    def __neg__(self):
        """Return the negation of the vector."""
        return Vector3D(-self._coords)

    def __add__(self, other):
        """Add another vector or a scalar to this vector."""
        if isinstance(other, Vector3D):
            return Vector3D(self._coords + other._coords)
        else:
            return Vector3D(self._coords + np.array(other))

    def __sub__(self, other):
        """Subtract another vector or a scalar from this vector."""
        if isinstance(other, Vector3D):
            return Vector3D(self._coords - other._coords)
        else:
            return Vector3D(self._coords - np.array(other))

    def __mul__(self, other):
        """Compute the dot product with another vector."""
        return np.dot(self._coords, other._coords)

    def __truediv__(self, scalar):
        """Divide the vector by a scalar."""
        return Vector3D(self._coords / scalar)

    def __pow__(self, other):
        """Compute the cross product with another vector or multiply by a scalar."""
        if isinstance(other, Vector3D):
            return Vector3D(np.cross(self._coords, other._coords))
        else:
            return Vector3D(self._coords * other)

    def __getitem__(self, index):
        """Get the coordinate at the specified index."""
        return self._coords[index]

    def __setitem__(self, index, value):
        """Set the coordinate at the specified index."""
        self._coords[index] = value

    def __contains__(self, value):
        """Check if a value is in the vector."""
        return value in self._coords

    def magnitude(self):
        """Return the magnitude (norm) of the vector."""
        return np.linalg.norm(self._coords)

    def magnitude_squared(self):
        """Return the squared magnitude of the vector."""
        return np.dot(self._coords, self._coords)

    def normalize(self):
        """Normalize the vector in place."""
        norm = self.magnitude()
        if norm != 0:
            self._coords /= norm

    def normalized(self):
        """Return a normalized copy of the vector."""
        norm = self.magnitude()
        if norm == 0:
            return Vector3D(self._coords)
        return Vector3D(self._coords / norm)

    def angle_with(self, other):
        dot_product = self * other
        magnitudes = self.magnitude() * other.magnitude()
        cos_angle = dot_product / magnitudes
        angle_radians = math.acos(cos_angle)
        return math.degrees(angle_radians)  # Convert radians to degrees

    def get_array(self):
        """Return a copy of the vector's coordinates as a numpy array."""
        return np.copy(self._coords)

    def left_multiply(self, matrix):
        """Multiply this vector by a matrix from the left."""
        return Vector3D(np.dot(matrix, self._coords))

    def right_multiply(self, matrix):
        """Multiply this vector by a matrix from the right."""
        return Vector3D(np.dot(self._coords, matrix))

    def copy(self):
        """Return a copy of this vector."""
        return Vector3D(self._coords)


def calc_angle(v1: Vector3D, v2: Vector3D, v3: Vector3D) -> float:
    """Calculate the angle between three vectors representing three connected points.

    This function calculates the angle formed by the vectors (v1-v2) and (v3-v2).

    :param v1: The first point that defines the angle.
    :param v2: The second point that defines the angle.
    :param v3: The third point that defines the angle.
    :type v1: Vector3D
    :type v2: Vector3D
    :type v3: Vector3D

    :return: The angle between the three vectors in degrees.
    :rtype: float

    Example:
    >>> v1 = Vector3D(1.0, 0.0, 0.0)
    >>> v2 = Vector3D(0.0, 1.0, 0.0)
    >>> v3 = Vector3D(0.0, 0.0, 1.0)
    >>> angle = calc_angle(v1, v2, v3)
    >>> print(float(round(angle, 2)))
    60.0
    """
    vec1 = v1 - v2
    vec3 = v3 - v2
    return vec1.angle_with(vec3)


def Ocalc_dihedral(v1: Vector3D, v2: Vector3D, v3: Vector3D, v4: Vector3D) -> float:
    """Calculate the dihedral angle between vectors v1-v2 and v4-v3.

    The dihedral angle is the angle between the planes formed by the vectors
    (v1-v2) and (v4-v3). The angle is in the range ]-pi, pi].

    :param v1: The first vector.
    :param v2: The second vector.
    :param v3: The third vector.
    :param v4: The fourth vector.
    :type v1: Vector3D
    :type v2: Vector3D
    :type v3: Vector3D
    :type v4: Vector3Dxk

    :return: The dihedral angle in radians.
    :rtype: float

    Example:
    >>> v1 = Vector3D(1.0, 0.0, 0.0)
    >>> v2 = Vector3D(0.0, 0.0, 0.0)
    >>> v3 = Vector3D(0.0, 1.0, 0.0)
    >>> v4 = Vector3D(0.0, 1.0, 1.0)
    >>> angle = float(calc_dihedral(v1, v2, v3, v4))
    >>> print(angle)
    90.0
    """
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = ab**cb
    v = db**cb
    w = u**v
    angle = u.angle_with(v)
    # Determine sign of angle
    try:
        if cb.angle_with(w) > 0.001:
            angle = -angle
    except ZeroDivisionError:
        pass
    return angle


def calc_dihedral(v1, v2, v3, v4):
    """
    Calculate the dihedral angle between four Vector3D objects.

    The dihedral angle is the angle between two planes formed by four points in space.
    This function calculates the angle in degrees.

    Parameters:
    v1, v2, v3, v4 (Vector3D): Four points in space, where:
        - v1, v2, v3 define the first plane.
        - v2, v3, v4 define the second plane.

    Returns:
    float: The dihedral angle in degrees.

    Example:
    >>> v1 = Vector3D(1.0, 1.0, 1.0)
    >>> v2 = Vector3D(1.0, 2.0, 1.0)
    >>> v3 = Vector3D(2.0, 2.0, 1.0)
    >>> v4 = Vector3D(2.0, 3.0, 1.0)
    >>> float(round(calc_dihedral(v1, v2, v3, v4), 2))
    0.0
    """
    p0 = v1.get_array()
    p1 = v2.get_array()
    p2 = v3.get_array()
    p3 = v4.get_array()

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.arctan2(y, x)

    return np.degrees(angle)


if __name__ == "__main__":
    import doctest

    doctest.testmod()


# end of file
