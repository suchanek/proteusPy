"""
Functions to create Disulfide Bond structural classes based on
dihedral angle rules. This module is part of the proteusPy package.
Many of the plotting functions have been folded into the DisulfideClassConstructor
object.

Author: Eric G. Suchanek, PhD.

(c) 2024 Eric G. Suchanek, PhD., All Rights Reserved
License: BSD
Last Modification: 2025-01-16 14:08:47 -egs-

"""

# pylint: disable=C0301
# pylint: disable=C0103

import math

import pandas as pd


def angle_within_range(angle, min_angle, max_angle):
    """
    Check whether the given angle is within the specified range.

    Parameters:
        angle (float): The angle to check, in degrees.
        min_angle (float): The minimum angle in the range, in degrees.
        max_angle (float): The maximum angle in the range, in degrees.

    Returns:
        bool: True if the angle is within the range, False otherwise.
    """

    # Convert angles to radians
    angle_rad = math.radians(angle)
    min_angle_rad = math.radians(min_angle)
    max_angle_rad = math.radians(max_angle)

    # Check whether the angle is within the range
    if min_angle_rad <= angle_rad <= max_angle_rad:
        return True
    else:
        return False


def get_quadrant(angle_deg):
    """
    Return the quadrant in which an angle in degrees lies.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The quadrant number (1, 2, 3, or 4) that the angle belongs to.
    """
    if angle_deg >= 0 and angle_deg < 90:
        return str(1)
    elif angle_deg >= 90 and angle_deg < 180:
        return str(2)
    elif angle_deg >= -180 and angle_deg < -90:
        return str(3)
    elif angle_deg >= -90 and angle_deg < 0:
        return str(4)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-180, 180).")


def filter_by_percentage(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """
    Filter a pandas DataFrame by percentage.

    :param df: A Pandas DataFrame with an 'percentage' column to filter by
    :param cutoff: A numeric value specifying the minimum incidence required for a row to be included in the output
    :type df: pandas.DataFrame
    :type cutoff: float
    :return: A new Pandas DataFrame containing only rows where the incidence is greater than or equal to the cutoff
    :rtype: pandas.DataFrame
    """
    return df[df["percentage"] >= cutoff]

def is_between(x, a, b):
    """
    Returns True if x is between a and b (inclusive), False otherwise.

    :param x: The input number to be tested.
    :type x: int or float
    :param a: The lower limit of the range to check against.
    :type a: int or float
    :param b: The upper limit of the range to check against.
    :type b: int or float
    :return: True if x is between a and b (inclusive), False otherwise.
    :rtype: bool
    """
    return a <= x <= b


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
