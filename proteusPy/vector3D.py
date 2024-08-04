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

import logging
import math

import numpy as np

from proteusPy.logger_config import get_logger

_logger = get_logger(__name__)  ## vector3d


class Vector3D:
    """
    A class to represent a three-dimensional vector.

    Attributes:
    -----------
    x : float
        The x-coordinate of the vector.
    y : float
        The y-coordinate of the vector.
    z : float
        The z-coordinate of the vector.

    Methods:
    --------
    __init__(x, y, z):
        Initializes the vector with x, y, and z coordinates.
    __add__(other):
        Adds another Vector3D to this vector and returns the result.
    __sub__(other):
        Subtracts another Vector3D from this vector and returns the result.
    __truediv__(scalar):
        Divides the vector by a scalar and returns the result.
    dot(other):
        Computes the dot product of this vector and another Vector3D.
    magnitude():
        Computes the magnitude (length) of the vector.
    normalize():
        Normalizes the vector (makes it unit length).
    normalized():
        Returns a normalized copy of the vector.
    angle_with(other):
        Computes the angle between this vector and another Vector3D in degrees.
    cross(other):
        Computes the cross product of this vector and another Vector3D.
    __repr__():
        Returns a string representation of the vector.

    Examples:
    ---------
    >>> v1 = Vector3D(1, 2, 3)
    >>> v2 = Vector3D(4, 5, 6)
    >>> v3 = v1 + v2
    >>> v3
    <Vector3D (5.00, 7.00, 9.00)>

    >>> v4 = v1 - v2
    >>> v4
    <Vector3D (-3.00, -3.00, -3.00)>

    >>> v5 = v1 / 2
    >>> v5
    <Vector3D (0.50, 1.00, 1.50)>

    >>> dot_product = v1 * v2
    >>> float(dot_product)
    32.0

    >>> magnitude = v1.magnitude()
    >>> float(round(magnitude, 2))
    3.74

    >>> angle = v1.angle_with(v2)
    >>> round(angle, 2)
    12.93

    >>> v6 = v1 ** v2
    >>> v6
    <Vector3D (-3.00, 6.00, -3.00)>
    """

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
        return f"<Vector3D ({x:.2f}, {y:.2f}, {z:.2f})>"

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

    def normalized(self):
        """Return a normalized copy of the Vector.

        To avoid allocating new objects use the ``normalize`` method.
        """
        v = self.copy()
        v.normalize()
        return v

    def normalize(self):
        """Normalize the vector in place."""
        norm = self.magnitude()
        if norm != 0:
            self._coords = self._coords / norm
        else:
            _logger.error("Vector3D: Cannot normalize a zero vector")

    def normalized(self):
        """Return a normalized copy of the vector."""
        res = self.copy()
        res.normalize()
        return res

    def angle_with(self, other):
        """
        Calculate the angle between this vector and another vector in degrees.

        This method computes the angle between the current vector and another vector
        using the dot product and magnitudes of the vectors. The result is returned
        in degrees.

        Parameters:
        other (Vector3D): The other vector to calculate the angle with.

        Returns:
        float: The angle between the two vectors in degrees.

        Example:
        >>> v1 = Vector3D(1, 0, 0)
        >>> v2 = Vector3D(0, 1, 0)
        >>> v1.angle_with(v2)
        90.0
        """
        dot_product = self * other
        magnitudes = self.magnitude() * other.magnitude()
        if magnitudes == 0:
            _logger.warning("Vector3D: Cannot calculate angle with zero vector")
            cos_angle = dot_product
        else:
            cos_angle = dot_product / magnitudes

        cos_theta = np.clip(cos_theta, -1.0, 1.0)
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


def calc_dihedral(v1: Vector3D, v2: Vector3D, v3: Vector3D, v4: Vector3D) -> float:
    """
    Return the dihedral angle between four 3D points, (-180-180 degrees).

    :param v1: The first point.
    :type v1: Vector3D
    :param v2: The second point.
    :type v2: Vector3D
    :param v3: The third point.
    :type v3: Vector3D
    :param v4: The fourth point.
    :type v4: Vector3D
    :return: The dihedral angle in degrees.
    :rtype: float

    :example:
    >>> v1 = Vector3D(0.0, 0.0, 0.0)
    >>> v2 = Vector3D(1.0, 0.0, 0.0)
    >>> v3 = Vector3D(1.0, 1.0, 1.0)
    >>> v4 = Vector3D(1.0, 1.0, 2.0)
    >>> float(round(calc_dihedral(v1, v2, v3, v4), 2))
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


def distance3d(p1: Vector3D, p2: Vector3D) -> float:
    """
    Calculate the 3D Euclidean distance for 2 Vector3D objects

    :param Vector p1: Point1
    :param Vector p2: Point2
    :return float distance: Distance between two points, Å

    Example:
    >>> from proteusPy import distance3d, Vector3D
    >>> p1 = Vector3D(1, 0, 0)
    >>> p2 = Vector3D(0, 1, 0)
    >>> distance3d(p1,p2)
    1.4142135623730951
    """

    _p1 = p1.get_array()
    _p2 = p2.get_array()
    if len(_p1) != 3 or len(_p2) != 3:
        _logger.error("distance3d() requires vectors of length 3!")
    d = math.dist(_p1, _p2)
    return d


def rms_difference(
    calculated_angles: np.ndarray, idealized_angles: np.ndarray
) -> float:
    """
    Calculate the Root Mean Square (RMS) difference between calculated angles and idealized angles.

    Parameters:
    calculated_angles (np.ndarray): An array of calculated angles.
    idealized_angles (np.ndarray): An array of idealized angles.

    Returns:
    float: The RMS difference between the calculated and idealized angles.

    Raises:
    ValueError: If the input arrays do not have the same shape or are empty.
    """
    if calculated_angles.shape != idealized_angles.shape:
        raise ValueError("Input arrays must have the same shape")

    if calculated_angles.size == 0 or idealized_angles.size == 0:
        raise ValueError("Input arrays must not be empty")

    differences = np.subtract(calculated_angles, idealized_angles)
    squared_differences = np.square(differences)
    mean_squared_difference = np.mean(squared_differences)
    rms_diff = np.sqrt(mean_squared_difference)
    return rms_diff

def calculate_bond_angle(a, b, c):
    """Calculate the bond angle between three points a, b, and c (in 3D space)."""
    ab = np.subtract(b, a)
    bc = np.subtract(c, b)
    
    ab_norm = np.linalg.norm(ab)
    bc_norm = np.linalg.norm(bc)
    
    # Check for zero-length vectors to avoid division by zero
    if ab_norm == 0 or bc_norm == 0:
        raise ValueError("One of the vectors has zero length, cannot calculate angle.")
    
    ab_unit = ab / ab_norm
    bc_unit = bc / bc_norm
    cos_theta = np.dot(ab_unit, bc_unit)
    
    # Ensure the value is within the valid range for arccos due to floating-point precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    
    return np.degrees(angle)

def compare_bond_angles(atom_coordinates):
    """
    Calculate all bond angles for a disulfide bond and compare them to idealized angles.

    Parameters:
    atom_coordinates (np.array): Array containing coordinates of atoms in the order:
                                 N1, CA1, C1, CB1, SG1, N2, CA2, C2, CB2, SG2

    Returns:
    float: RMS difference between calculated bond angles and idealized bond angles.
    """
    idealized_angles = {
        ("N1", "CA1", "C1"): 111,
        ("CA1", "C1", "CB1"): 109,
        ("C1", "CB1", "SG1"): 114,
        ("CB1", "SG1", "SG2"): 102,  # This angle is for the disulfide bond itself
        ("SG1", "SG2", "CB2"): 102,  # This angle is for the disulfide bond itself
        ("CB2", "C2", "CA2"): 114,
        ("C2", "CA2", "N2"): 109,
        ("CA2", "N2", "N1"): 111,
    }

    # List of triplets for which we need to calculate bond angles
    angle_triplets = [
        ("N1", "CA1", "C1"),
        ("CA1", "C1", "CB1"),
        ("C1", "CB1", "SG1"),
        (
            "CB1",
            "SG1",
            "SG2",
        ),  # Note: we need to handle the disulfide bond case separately
        ("SG1", "SG2", "CB2"),
        ("CB2", "C2", "CA2"),
        ("C2", "CA2", "N2"),
        ("CA2", "N2", "N1"),
    ]

    atom_indices = {
        "N1": 0,
        "CA1": 1,
        "C1": 2,
        "CB1": 3,
        "SG1": 4,
        "SG2": 9,
        "CB2": 8,
        "C2": 7,
        "CA2": 6,
        "N2": 5,
    }

    calculated_angles = []
    for triplet in angle_triplets:
        a = atom_coordinates[atom_indices[triplet[0]]]
        b = atom_coordinates[atom_indices[triplet[1]]]
        c = atom_coordinates[atom_indices[triplet[2]]]
        try:
            angle = calculate_bond_angle(a, b, c)
        except ValueError as e:
            print(f"Error calculating angle for atoms {triplet}: {e}")
            return None
        calculated_angles.append(angle)

    # Convert idealized angles to a list
    idealized_angles_list = [idealized_angles[triplet] for triplet in angle_triplets]

    # Calculate RMS difference
    rms_diff = rms_difference(calculated_angles, idealized_angles_list)
    return rms_diff


if __name__ == "__main__":
    import doctest

    doctest.testmod()


# end of file
