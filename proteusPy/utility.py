'''
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: MIT\n
Copyright (c)2023 Eric G. Suchanek, PhD, all rights reserved
'''

# Last modification 2/18/23 -egs-

import math
import numpy as np

import copy
import subprocess

import os
import requests

from numpy import linspace
from matplotlib import cm

import proteusPy
from proteusPy.proteusPyWarning import ProteusPyWarning
# from proteusPy.DisulfideList import DisulfideList

from Bio.PDB.vectors import Vector
from Bio.PDB import PDBParser

def distance_squared(p1: np.array, p2: np.array) -> np.array:
    '''
    Returns the square of the N-dimensional distance between the
    two arrays.

    :param np.array p1: N-dimensional array 1
    :param np.array p2: N-dimensional array 2
    :return: np.array N-dimensional distance squared Å^2

    Example:

    >>> from proteusPy.utility import distance_squared
    >>> p1 = np.array([1.0, 0.0, 0.0])
    >>> p2 = np.array([0, 1.0, 0])
    >>> distance_squared(p1, p2)
    2.0
    '''
    return np.sum(np.square(np.subtract(p1, p2)))

def distance3d(p1: Vector, p2: Vector) -> float:
    '''
    Calculate the 3D Euclidean distance for 2 Vector objects
    
    :param Vector p1: Point1
    :param Vector p2: Point2
    :return float distance: Distance between two points, Å

    Example:

    >>> from proteusPy.utility import distance3d
    >>> p1 = Vector(1, 0, 0)
    >>> p2 = Vector(0, 1, 0)
    >>> distance3d(p1,p2)
    1.4142135623730951
    '''

    _p1 = p1.get_array()
    _p2 = p2.get_array()
    if (len(_p1) != 3 or len(_p2) != 3):
        raise ProteusPyWarning("distance3d() requires vectors of length 3!")
    d = math.dist(_p1, _p2)
    return d

def cmap_vector(steps):
    '''
    Return an RGB array of steps rows using the ```jet``` colormap.
    
    :param int steps: number of RGB elements to return
    :return: np.array [steps][3] array of RGB values.

    Example:
    >>> from proteusPy.utility import cmap_vector
    >>> cmap_vector(12)
    array([[  0.,   0., 127.],
           [  0.,   0., 232.],
           [  0.,  56., 255.],
           [  0., 148., 255.],
           [ 12., 244., 234.],
           [ 86., 255., 160.],
           [160., 255.,  86.],
           [234., 255.,  12.],
           [255., 170.,   0.],
           [255.,  85.,   0.],
           [232.,   0.,   0.],
           [127.,   0.,   0.]])

    '''

    rgbcol = np.zeros(shape=(steps, 3))
    norm = linspace(0.0, 1.0, steps)

    # colormap possible values = viridis, jet, spectral
    rgb_all = cm.jet(norm, bytes=True) 
    i = 0
    
    for rgb in rgb_all:
        rgbcol[i][0] = rgb[0]
        rgbcol[i][1] = rgb[1]
        rgbcol[i][2] = rgb[2]
        i += 1
    return rgbcol

def get_jet_colormap(steps):
    """
    Returns an array of uniformly spaced RGB values using the 'jet' colormap.

    :param steps: The number of steps in the output array.

    :return: An array of uniformly spaced RGB values using the 'jet' colormap. The shape of the array is (steps, 3).
    :rtype: numpy.ndarray

    :example:
        >>> get_jet_colormap(5)
        array([[  0,   0, 128],
               [  0, 128, 255],
               [128, 255, 128],
               [255, 255,   0],
               [255, 128,   0]])
    """
    norm = np.linspace(0.0, 1.0, steps)
    colormap = cm.get_cmap('jet', steps)
    rgbcol = colormap(norm, bytes=True)[:,:3]

    return rgbcol


  
def grid_dimensions(n):
    '''
    Calculate rows and columns for the given needed to display
    a given number of disulfides in a square aspect.

    :param n: Number of Disulfides
    :return: int rows, columns
    '''
    
    root = math.sqrt(n)
    # If the square root is a whole number, return that as the number of rows and columns
    if root == int(root):
        return int(root), int(root)
    # If the square root is not a whole number, round up and return that as the number of columns
    # and calculate the number of rows as the number of images divided by the number of columns
    else:
        columns = math.ceil(root)
        return int(n / columns), int(columns)

def Check_chains(pdbid, pdbdir, verbose=True):
    '''
    Returns True if structure has multiple chains of identical length,
    False otherwise. Primarily internal use.

    :param pdbid: PDBID identifier
    :param pdbdir: PDB directory containing structures
    :param verbose: Verbosity, defaults to True
    '''
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(pdbid, file=f'{pdbdir}pdb{pdbid}.ent')
    
    # dictionary of tuples with SSBond prox and distal
    ssbond_dict = structure.header['ssbond']
    
    if verbose:
        print(f'ssbond dict: {ssbond_dict}')

    same = False
    model = structure[0]
    chainlist = model.get_list()

    if len(chainlist) > 1:
        chain_lens = []
        if verbose:
            print(f'multiple chains. {chainlist}')
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                print(f'Chain: {chain_id}, length: {chain_length}')
            chain_lens.append(chain_length)

        if np.min(chain_lens) != np.max(chain_lens):
            same = False
            if verbose:
                print(f'chain lengths are unequal: {chain_lens}')
        else:
            same = True
            if verbose:
                print(f'Chains are equal length, assuming the same. {chain_lens}')
    return(same)

# given the full dictionary, walk through all the keys (PDB ID)
# for each PDB_ID SS list, find and extract the SS for the first chain
# update the 'pruned' dict with the now shorter SS list

def extract_firstchain_ss(sslist, verbose=False):
    '''
    Function extracts disulfides from the first chain found in
    the SSdict, returns them as a DisulfideList along with the
    number of Xchain disulfides.

    :param sslist: Starting SS list
    :return: (Pruned SS list, xchain) 
    '''

    chainlist = []
    pc = dc = ''
    res = proteusPy.DisulfideList.DisulfideList([], sslist.id)
    xchain = 0

    # build list of chains
    for ss in sslist:
        pc = ss.proximal_chain
        dc = ss.distal_chain
        if pc != dc:
            xchain += 1
            if verbose:
                print(f'--> extract_firstchain_ss(): Cross chain ss: {ss}')
        chainlist.append(pc)
    chain = chainlist[0]

    for ss in sslist:
        if ss.proximal_chain == chain:
            res.append(ss)
    
    return res, xchain

def prune_extra_ss(sslist):
    '''
    Given a dict of disulfides, check for extra chains, grab only the disulfides from
    the first chain and return a dict containing only the first chain disulfides

    :param ssdict: input dictionary with disulfides
    '''
    xchain = 0

    #print(f'Processing: {ss} with: {sslist}')
    id = sslist.pdb_id
    pruned_list = proteusPy.DisulfideList.DisulfideList([], id)
    pruned_list, xchain = extract_firstchain_ss(sslist)
        
    return copy.deepcopy(pruned_list), xchain

def download_file(url, directory):
    '''
    Download the given URL to the input directory

    :param url: File URL
    :param directory: Directory path for saving.
    '''
    file_name = url.split("/")[-1]
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        command = ["wget", "-P", directory, url]
        subprocess.run(command, check=True)
        print("Download complete.")
    else:
        print(f"{file_name} already exists in {directory}.")

import pandas as pd

def add_sign_columns(df):
    """
    Create new columns with the sign of each dehdral angle (chi1-chi5)
    column and return a new DataFrame with the additional columns.
    This is used to build disulfide classes.

    :param df: pandas.DataFrame - The input DataFrame containing 
    the dihedral angle (chi1-chi5) columns.
        
    :return: A new DataFrame containing the columns 'ss_id', 'chi1_s', 
    'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s' which represent the signs of 
    the dihedral angle columns in the input DataFrame.
        
    :example:
    >>> import pandas as pd
    >>> data = {'ss_id': [1, 2, 3], 'chi1': [-2, 1.0, 1.3], 'chi2': [0.8, -1.5, 0], 
    ...         'chi3': [-1, 2, 0.1], 'chi4': [0, 0.9, -1.1], 'chi5': [0.2, -0.6, -0.8]}
    >>> df = pd.DataFrame(data)
    >>> res_df = add_sign_columns(df)
    >>> print(res_df)
       ss_id  chi1_s  chi2_s  chi3_s  chi4_s  chi5_s
    0      1      -1       1      -1       1       1
    1      2       1      -1       1       1      -1
    2      3       1       1       1      -1      -1
    """
    # Create columns for the resulting DF
    tors_vector_cols = ['ss_id', 'chi1_s', 'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s']
    res_df = pd.DataFrame(columns=tors_vector_cols)
    
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)
    res_df = df[tors_vector_cols].copy()
    return res_df


def group_by_sign(df):
    '''
    Group a DataFrame by the sign of each dihedral angle (chi1-chi5) column.

    This function creates new columns in the input DataFrame with the sign of each chi column, 
    and groups the DataFrame by these new columns. The function returns the aggregated data, including 
    the mean and standard deviation of the 'ca_distance', 'torsion_length', and 'energy' columns.

    :param df: The input DataFrame to group by sign.
    :type df: pandas.DataFrame
    :return: The DataFrame grouped by sign, including means and standard deviations.
    :rtype: pandas.DataFrame

    :example:
    >>> df = pd.DataFrame({'pdbid': ['1ABC', '1DEF', '1GHI', '1HIK'],
    ...                    'chi1': [120.0, -45.0, 70.0, 90],
    ...                    'chi2': [90.0, 180.0, -120.0, -90],
    ...                    'chi3': [-45.0, -80.0, 20.0, 0],
    ...                    'chi4': [0.0, 100.0, -150.0, -120.0],
    ...                    'chi5': [-120.0, -10.0, 160.0, -120.0],
    ...                    'ca_distance': [3.5, 3.8, 2.5, 3.3],
    ...                    'torsion_length': [3.2, 2.8, 3.0, 4.4],
    ...                    'energy': [-12.0, -10.0, -15.0, -20.0]})
    >>> grouped = group_by_sign(df)
    >>> grouped
       chi1_s  chi2_s  chi3_s  chi4_s  chi5_s  ca_distance_mean  ca_distance_std  torsion_length_mean  torsion_length_std  energy_mean  energy_std
    0      -1       1      -1       1      -1               3.8              NaN                  2.8                 NaN        -10.0         NaN
    1       1      -1       1      -1      -1               3.3              NaN                  4.4                 NaN        -20.0         NaN
    2       1      -1       1      -1       1               2.5              NaN                  3.0                 NaN        -15.0         NaN
    3       1       1      -1       1      -1               3.5              NaN                  3.2                 NaN        -12.0         NaN
    
    '''
    
    
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)

    # Group the DataFrame by the sign columns and return the aggregated data
    group_columns = sign_columns
    agg_columns = ['ca_distance', 'torsion_length', 'energy']
    grouped = df.groupby(group_columns)[agg_columns].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()
   

def Create_classes(df):
    """
    Group the DataFrame by the sign of the chi columns and create a new class ID column for each unique grouping.

    :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'torsion_length', and 'energy'.
    :return: A pandas DataFrame containing columns 'class_id', 'ss_id', and 'count', where 'class_id' is a unique identifier for each grouping of chi signs, 'ss_id' is a list of all 'ss_id' values in that grouping, and 'count' is the number of rows in that grouping.
    :example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'ss_id': [1, 2, 3, 4, 5],
    ...    'chi1': [1.0, -1.0, 1.0, 1.0, -1.0],
    ...    'chi2': [-1.0, -1.0, -1.0, 1.0, 1.0],
    ...    'chi3': [-1.0, 1.0, -1.0, 1.0, -1.0],
    ...    'chi4': [1.0, -1.0, 1.0, -1.0, 1.0],
    ...    'chi5': [1.0, -1.0, -1.0, -1.0, -1.0],
    ...    'ca_distance': [3.1, 3.2, 3.3, 3.4, 3.5],
    ...    'torsion_length': [120.1, 120.2, 120.3, 120.4, 121.0],
    ...    'energy': [-2.3, -2.2, -2.1, -2.0, -1.9]
    ... })
    >>> Create_classes(df)
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
