'''
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: MIT\n
Copyright (c)2023 Eric G. Suchanek, PhD, all rights reserved
'''

# Last modification 2/18/23 -egs-

__pdoc__ = {'__all__': True}

import math
import numpy as np

import copy
import subprocess
import pickle

import os
import requests

from numpy import linspace
from matplotlib import cm

import proteusPy
from proteusPy.proteusPyWarning import ProteusPyWarning

from Bio.PDB.vectors import Vector
from Bio.PDB import PDBParser

import pandas as pd
from proteusPy.data import DATA_DIR

def distance_squared(p1: np.array, p2: np.array) -> np.array:
    '''
    Returns the square of the N-dimensional distance between the
    two arrays.

    :param np.array p1: N-dimensional array 1
    :param np.array p2: N-dimensional array 2
    :return: np.array N-dimensional distance squared Å^2

    Example
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

def get_jet_colormap(steps):
    """
    Returns an array of uniformly spaced RGB values using the 'jet' colormap.

    :param steps: The number of steps in the output array.

    :return: An array of uniformly spaced RGB values using the 'jet' colormap. The shape of the array is (steps, 3).
    :rtype: numpy.ndarray

    Example:
        >>> get_jet_colormap(5)
        array([[  0,   0, 127],
               [  0, 127, 255],
               [123, 255, 123],
               [255, 151,   0],
               [127,   0,   0]], dtype=uint8)
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

import psutil

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def print_memory_used():
    '''
    Print memory used by the proteusPy process.
    '''
    mem = get_memory_usage() / (1024 ** 3) # to GB

    print(f'--> proteusPy {proteusPy.__version__}: Memory Used: {mem:.2f} GB')

def About_proteusPy():
    """
    *proteusPy* is a Python package specializing in the modeling and analysis of 
    proteins of known structure with an emphasis on Disulfide Bonds. This package 
    reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), 
    and relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) 
    class. The turtle implements the functions `Move`, `Roll`, `Yaw`, `Pitch` 
    and `Turn` for movement in a three-dimensional space. The 
    [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class 
    implements methods to analyze the protein structure stabilizing element known as a 
    *Disulfide Bond*. This class and its underlying methods are being used to perform a 
    structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB 
    protein data bank.

    ### Virtual Environment Installation/Creation

    1. Install Anaconda (<http://anaconda.org>)
       - Create a new environment using python 3.9
       - Activate the environment
    2. Build the environment. At this point it's probably best to clone the repo via github 
       since it contains all of the notebooks and test programs. Ultimately the distribution 
       can be used from pyPi as a normal package.
       - Using pyPi:
         - python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ proteusPy
       - From gitHub:
         - Install git-lfs
           - https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage
           - From a shell prompt: git-lfs track "*.csv" "*.pkl" "*.mp4"
         - git clone https://github.com/suchanek/proteusPy/proteusPy.git
         - cd into the repo
         - pip install .

    #### Publications
    * https://doi.org/10.1021/bi00368a023
    * https://doi.org/10.1021/bi00368a024
    * https://doi.org/10.1016/0092-8674(92)90140-8
    * http://dx.doi.org/10.2174/092986708783330566

    *NB:* This distribution is actively being developed and will be difficult to implement 
    locally unless the BioPython patch is applied. Also, if you're running on an M-series Mac 
    then it's important to install Biopython first, since the generic release won't build 
    on the M1. 2/18/23 -egs

    Eric G. Suchanek, PhD. mailto:suchanek@mac.com
    """
    return About_proteusPy.__doc__

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
