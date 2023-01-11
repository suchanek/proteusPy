#
# init for proteusPy package
# Copyright (c) 2022 Eric G. Suchanek, PhD., all rights reserved
# Subject to the GNU public license.
# Cα Cβ Sγ Χ1 - Χ5 Χ

__version__ = "0.8dev"

import sys
import os
import glob
import warnings
import copy
import shutil

import pickle
import time
import datetime
import math
import numpy

import pandas as pd
from tqdm import tqdm
from numpy import cos

from Bio.PDB.vectors import calc_dihedral, calc_angle

from .proteusGlobals import *
from .proteusPyWarning import *

from .DisulfideGlobals import *
from .atoms import *
from .DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException

from .turtle3D import Turtle3D
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN
from .residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

#from .Disulfide import Disulfide
from .Disulfide import todeg, torad, distance3d
from .Disulfide import Download_Disulfides, Extract_Disulfides
from .Disulfide import  Check_chains, Distance_RMS, Torsion_RMS

#from .Disulfide import DisulfideList
from .DisulfideLoader import DisulfideLoader
from Bio.PDB import Select

from matplotlib import cm
from numpy import linspace

class CysSelect(Select):
    def accept_residue(self, residue):
        if residue.get_name() == 'CYS':
            return True
        else:
            return False


def cmap_vector(steps):
    '''
    Return an RGB array of steps rows
    
    Argument:
        steps: number of RGB elements to return

    Returns: 
        steps X 3 array of RGB values.
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
    # Calculate the square root of the number of images
    root = math.sqrt(n)
    # If the square root is a whole number, return that as the number of rows and columns
    if root == int(root):
        return int(root), int(root)
    # If the square root is not a whole number, round up and return that as the number of columns
    # and calculate the number of rows as the number of images divided by the number of columns
    else:
        columns = math.ceil(root)
        return int(n / columns), int(columns)

# end of file
