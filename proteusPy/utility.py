'''
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: MIT\n
Copyright (c)2023 Eric G. Suchanek, PhD, all rights reserved
'''

# Last modification 1/22/23 -egs-

import math
import numpy
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

def distance_squared(p1: numpy.array, p2: numpy.array) -> numpy.array:
    '''
    Returns the square of the N-dimensional distance between the
    two arrays.

    :param numpy.array p1: N-dimensional array 1
    :param numpy.array p2: N-dimensional array 2
    :return: numpy.array N-dimensional distance squared Å^2

    Example:

    >>> from proteusPy.utility import distance_squared
    >>> p1 = numpy.array([1.0, 0.0, 0.0])
    >>> p2 = numpy.array([0, 1.0, 0])
    >>> distance_squared(p1, p2)
    2.0
    '''
    return numpy.sum(numpy.square(numpy.subtract(p1, p2)))

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
    :return: numpy.array [steps][3] array of RGB values.

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

    rgbcol = numpy.zeros(shape=(steps, 3))
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
   
def grid_dimensions(n):
    '''
    Calculate rows and columns for the given needed to display
    a given number of disulfides in a square aspect.

    :param n: Number of Disulfides
    :type n: int
    :return: int rows, columns
    :rtype: int
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

        if numpy.min(chain_lens) != numpy.max(chain_lens):
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

# ChatGPT input:
# write a python function that takes a
# pandas dataframe containing columns 'pdbid', 'chi1', 'chi2, 'chi3, 'chi4', 'chi5', 'ca_distance', 'torsion_length', 'energy'
# and then adds additional columns labeled 'chi1_s', 'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s' whose values are either -1 or 1 based
# on the sign of the 'chi1', 'chi2, 'chi3', 'chi4', 'chi5' columns

import pandas as pd

def add_sign_columns(df):
    # Create columns for the resulting DF
    tors_vector_cols = ['ss_id', 'chi1_s', 'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s']

    res_df = pd.DataFrame(columns=tors_vector_cols)
    
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)
    res_df = df[tors_vector_cols].copy()
    return res_df

'''
    let's continue and take the resulting data frame and group it by columns 
    'chi1_s', 'chi2_s', 'chi3_s', 'chi4_s', 'chi5_s' and return that as a new data frame
'''
    
def group_by_sign(df):
    # Create new columns with the sign of each chi column
    chi_columns = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']
    sign_columns = [col + '_s' for col in chi_columns]
    df[sign_columns] = df[chi_columns].applymap(lambda x: 1 if x >= 0 else -1)

    # Group the DataFrame by the sign columns and return the aggregated data
    group_columns = sign_columns
    agg_columns = ['ca_distance', 'torsion_length', 'energy']
    grouped = df.groupby(group_columns)[agg_columns].agg(['mean', 'std', 'count'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()

def create_classes(df):
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

    return grouped

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
