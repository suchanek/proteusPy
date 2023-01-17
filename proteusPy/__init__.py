# Initialization for the proteusPy package
# Copyright (c) 2023 Eric G. Suchanek, PhD., all rights reserved
# Subject to the MIT public license.

__version__ = "0.9dev"

import sys
import os
import glob
import warnings
import copy

import time
import datetime
import math

from matplotlib import cm
from numpy import linspace
import numpy

from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.PDB import Select, Vector

from .proteusPyWarning import ProteusPyWarning
from .ProteusGlobals import *

from .DisulfideExceptions import DisulfideIOException, DisulfideConstructionWarning, DisulfideConstructionException

from .turtle3D import Turtle3D
from .turtle3D import ORIENT_BACKBONE, ORIENT_SIDECHAIN
from .residue import build_residue, get_backbone_from_chain, to_alpha, to_carbonyl, to_nitrogen, to_oxygen

from .Disulfide import Download_Disulfides, Extract_Disulfides
from .Disulfide import Check_chains, Distance_RMS, Torsion_RMS

from .DisulfideLoader import DisulfideLoader
from .atoms import *


def torad(deg):
    return(numpy.radians(deg))

def todeg(rad):
    return(numpy.degrees(rad))

class CysSelect(Select):
    def accept_residue(self, residue):
        if residue.get_name() == 'CYS':
            return True
        else:
            return False

def distance3d(p1: Vector, p2: Vector):
    '''
    Calculate the 3D Euclidean distance for 2 Vector objects
    
    :param p1, p2: Vector objects of dimensionality 3
    :return distance: float Distance between two points

    Example:

    >>> from proteusPy import distance3d
    >>> from Bio.PDB import Vector
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
    Return an RGB array of steps rows using
    
    Argument:
        :param steps: int - number of RGB elements to return

    Returns: 
        :return: numpy.array [steps][3] array of RGB values.

    Example:
    >>> import numpy
    >>> from matplotlib import cm
    >>> from proteusPy import cmap_vector
    >>> cmap_vector(12)
    array([[ 31., 119., 180.],
       [174., 199., 232.],
       [255., 187., 120.],
       [152., 223., 138.],
       [255., 152., 150.],
       [197., 176., 213.],
       [140.,  86.,  75.],
       [227., 119., 194.],
       [127., 127., 127.],
       [188., 189.,  34.],
       [ 23., 190., 207.],
       [158., 218., 229.]])

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

# end of file
