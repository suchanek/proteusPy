'''
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: MIT\n
Copyright (c)2023 Eric G. Suchanek, PhD, all rights reserved
'''

# Last modification 1/22/23 -egs-

import math
import numpy
from numpy import linspace
from matplotlib import cm
from proteusPy.proteusPyWarning import ProteusPyWarning

from Bio.PDB.vectors import Vector

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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file

