'''
Functions to create Disulfide Bond structural classes based on
dihedral angle rules.

Author: Eric G. Suchanek, PhD. \n

(c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
License: MIT
Last Modification: 2/18/23

'''

import copy
from io import StringIO
import time
import datetime
import pickle

import pandas as pd
import numpy as np
import proteusPy

from proteusPy.data import DATA_DIR, SS_CLASS_DICT_FILE, CLASSOBJ_FNAME
from proteusPy.data import SS_CLASS_DEFINITIONS


def create_classes(df):
    """
    Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance', 'torsion_length', and 'energy'.
    :return: A pandas DataFrame containing columns 'class_id', 'ss_id', and 'count', where 'class_id' is a unique identifier for each grouping of chi signs, 'ss_id' is a list of all 'ss_id' values in that grouping, and 'count' is the number of rows in that grouping.
    
    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'ss_id': [1, 2, 3, 4, 5],
    ...    'chi1': [1.0, -1.0, 1.0, 1.0, -1.0],
    ...    'chi2': [-1.0, -1.0, -1.0, 1.0, 1.0],
    ...    'chi3': [-1.0, 1.0, -1.0, 1.0, -1.0],
    ...    'chi4': [1.0, -1.0, 1.0, -1.0, 1.0],
    ...    'chi5': [1.0, -1.0, -1.0, -1.0, -1.0],
    ...    'ca_distance': [3.1, 3.2, 3.3, 3.4, 3.5],
    ...    'cb_distance': [3.1, 3.2, 3.3, 3.4, 3.5],
    ...    'torsion_length': [120.1, 120.2, 120.3, 120.4, 121.0],
    ...    'energy': [-2.3, -2.2, -2.1, -2.0, -1.9]
    ... })
    >>> create_classes(df)
      class_id ss_id  count  incidence  percentage
    0    00200   [2]      1        0.2        20.0
    1    02020   [5]      1        0.2        20.0
    2    20020   [3]      1        0.2        20.0
    3    20022   [1]      1        0.2        20.0
    4    22200   [4]      1        0.2        20.0

    """
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)
    
    # Create a new column with the class ID for each row
    class_id_column = 'class_id'
    df[class_id_column] = (df[sign_columns] + 1).apply(lambda x: ''.join(x.astype(str)), axis=1)

    # Group the DataFrame by the class ID and return the grouped data
    grouped = df.groupby(class_id_column)['ss_id'].unique().reset_index()
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['ss_id'].apply(lambda x: len(x)/len(df))
    grouped['percentage'] = grouped['incidence'].apply(lambda x: 100 * x)

    return grouped

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
    import math
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

def get_sixth_quadrant(angle_deg):
    """
    Returns the sextant in which an angle in degrees lies if the area is described by dividing a unit circle into 6 equal segments.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The sextant (1-6) that the angle belongs to.
    """
    # Normalize the angle to the range [0, 360)
    angle_deg = angle_deg % 360

    if angle_deg >= 0 and angle_deg < 60:
        return str(1)
    elif angle_deg >= 60 and angle_deg < 120:
        return str(2)
    elif angle_deg >= 120 and angle_deg < 180:
        return str(3)
    elif angle_deg >= 180 and angle_deg < 240:
        return str(4)
    elif angle_deg >= 240 and angle_deg < 300:
        return str(5)
    elif angle_deg >= 300 and angle_deg < 360:
        return str(6)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-360, 360).")

def get_half_quadrant(angle_deg):
    """
    Returns the half-quadrant in which an angle in degrees lies.

    Parameters:
        angle_deg (float): The angle in degrees.

    Returns:
        int: The half-quadrant number (1-8) that the angle belongs to.
    """

    if angle_deg >= 0 and angle_deg < 45:
        return str(1)
    elif angle_deg >= 45 and angle_deg < 90:
        return str(2)
    elif angle_deg >= 90 and angle_deg < 135:
        return str(3)
    elif angle_deg >= 135 and angle_deg < 180:
        return str(4)
    elif angle_deg >= -45 and angle_deg < 0:
        return str(5)
    elif angle_deg >= -90 and angle_deg < -45:
        return str(6)
    elif angle_deg >= -135 and angle_deg < -90:
        return str(7)
    elif angle_deg >= -180 and angle_deg < -135:
        return str(8)
    else:
        raise ValueError("Invalid angle value: angle must be in the range [-180, 180).")

def create_six_class_df(df) -> pd.DataFrame:
    """
    Create a new DataFrame from the input with a 6-class encoding for input 'chi' values.
    
    The function takes a pandas DataFrame containing the following columns:
    'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance',
    'torsion_length', 'energy', and 'rho', and adds a class ID column based on the following rules:
    
    1. A new column named `class_id` is added, which is the concatenation of the individual class IDs per Chi.
    2. The DataFrame is grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
    the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
    percentage of incidence.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
               'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
    :return: The grouped DataFrame with the added class column.
    """
    
    _df = pd.DataFrame()
    # create the chi_t columns for each chi column
    for col_name in ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']:
        _df[col_name + '_t'] = df[col_name].apply(get_sixth_quadrant)
    
    # create the class_id column
    df['class_id'] = _df[['chi1_t', 'chi2_t', 'chi3_t', 'chi4_t', 'chi5_t']].apply(lambda x: ''.join(x), axis=1)

    # group the DataFrame by class_id and return the grouped data
    grouped = df.groupby('class_id').agg({'ss_id': 'unique'})
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['count'] / len(df)
    grouped['percentage'] = grouped['incidence'] * 100
    grouped.reset_index(inplace=True)

    return grouped

def create_quat_classes(df):
    """
    Add new columns to the input DataFrame with a 4-class encoding for input 'chi' values.

    Takes a DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance',
    'cb_distance', 'torsion_length', 'energy', and 'rho' and adds new columns based on the following rules:
    1. The 'chi_t' column is set to the quadrant in which the dihedral angle is located.
    
    A new column named `class_id` is also added, which is the concatenation of the `_t` columns. The DataFrame is then
    grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
    the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
    percentage of incidence.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
               'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
    :return: The input DataFrame with the added columns
    """

    new_cols = []
    for col_name in ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']:
        col = df[col_name]
        new_col = []
        for val in col:
            new_col.append(get_sixth_quadrant(val))
        new_col_name = col_name + '_t'
        new_cols.append(new_col_name)
        df[new_col_name] = new_col
    
    class_id_column = 'class_id'

    df['class_id'] = df[new_cols].apply(lambda x: ''.join(x), axis=1)

    # Group the DataFrame by the class ID and return the grouped data
    grouped = df.groupby(class_id_column)['ss_id'].unique().reset_index()
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['ss_id'].apply(lambda x: len(x)/len(df))
    grouped['percentage'] = grouped['incidence'].apply(lambda x: 100 * x)

    return grouped

def Ocreate_quat_classes(df):
    """
    Add new columns to the input DataFrame with a 4-class encoding for input 'chi' values.

    Takes a DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance',
    'cb_distance', 'torsion_length', 'energy', and 'rho' and adds new columns based on the following rules:
    1. If the 'chi' column is between -90 to -60 then the new column is '-'. (g-)
    2. If it's between 60 to 90, then the new column is '+'. (g+)
    3. If it's between -180 to -150 then the new column is '*'. (trans)
    4. If it's between 150 to 180 then the new column is '@'. (trans)
    5. Otherwise the new column is '!' 

    A new column named `class_id` is also added, which is the concatenation of the `_t` columns. The DataFrame is then
    grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
    the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
    percentage of incidence.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
               'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
    :return: The input DataFrame with the added columns
    """
    
    # - + * @ 
    new_cols = []
    for col_name in ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']:
        col = df[col_name]
        new_col = []
        for val in col:
            if is_between(val, -90, -60) or is_between(val, 60, 90):
                new_col.append('-')
            elif is_between(val, -181, -150) or is_between(val, 150, 180):
                new_col.append('*')
            elif is_between(val, -150, -120) or is_between(val, 120, 150):
                new_col.append('+')
            else:
                new_col.append('!')
        new_col_name = col_name + '_t'
        new_cols.append(new_col_name)
        df[new_col_name] = new_col
    
    class_id_column = 'class_id'

    df['class_id'] = df[new_cols].apply(lambda x: ''.join(x), axis=1)

    # Group the DataFrame by the class ID and return the grouped data
    grouped = df.groupby(class_id_column)['ss_id'].unique().reset_index()
    grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
    grouped['incidence'] = grouped['ss_id'].apply(lambda x: len(x)/len(df))
    grouped['percentage'] = grouped['incidence'].apply(lambda x: 100 * x)

    return grouped

def get_section(angle_deg, basis):
    """
    Returns the section in which an angle in degrees lies if the section is described by dividing a unit circle into `basis` equal segments.

    Parameters:
        angle_deg (float): The angle in degrees.
        basis (int): The number of equal angular divisions into which the unit circle is divided.

    Returns:
        int: The section number (1-basis) that the angle belongs to.
    """
    # Normalize the angle to the range [-180, 180)
    angle_deg = angle_deg % 360
    if angle_deg < -180:
        angle_deg += 360
    elif angle_deg >= 180:
        angle_deg -= 360

    # Calculate the size of each segment
    segment_size = 360 / basis

    # Calculate the section number
    section = int(angle_deg // segment_size) + 1
    if section <= 0:
        section += basis

    return str(section)

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
