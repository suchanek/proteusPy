#
# Turtle3D.py
#
# Implementation of a 3D Turtle in Python.
# Author: Eric G. Suchanek, PhD
# A part of the program Proteus, https://github.com/suchanek/proteus,
# a program for the manipulation and analysis of macromolecules
# Based on the C implementation originally authored by Eric G. Suchanek PhD 1990
# 

import numpy
import math
from Bio.PDB.vectors import Vector

_DOWN_ = -1
_UP_ = 1
_ORIENTATION_INIT = -1

# global variables are in globals.py
from globals import *

"""
    Return a (left multiplying) matrix that rotates p onto q.

    :param p: moving vector
    :type p: L{Vector}

    :param q: fixed vector
    :type q: L{Vector}

    :return: rotation matrix that rotates p onto q
    :rtype: 3x3 Numeric array

    Examples
    --------
    >>> from Bio.PDB.vectors import rotmat
    >>> p, q = Vector(1, 2, 3), Vector(2, 3, 5)
    >>> r = rotmat(p, q)
    >>> print(q)
    <Vector 2.00, 3.00, 5.00>
    >>> print(p)
    <Vector 1.00, 2.00, 3.00>
    >>> p.left_multiply(r)
    <Vector 1.21, 1.82, 3.03>

    """

# Class Definition begins

class Turtle3D:
    """3D Turtle."""

    def __init__(self, name="3D_Turtle"):
        # internal rep is arrays
        self._position = numpy.array((0.0, 0.0, 0.0), "d")
        self._heading = numpy.array((1.0, 0.0, 0.0), "d")
        self._left = numpy.array((0.0, 1.0, 0.0), "d")
        self._up = numpy.array((0.0, 0.0, 1.0), "d")

        # expose these as Vectors
        self.position = Vector(0.0, 0.0, 0.0)
        self.heading = Vector(1.0, 0.0, 0.0)
        self.left = Vector(0.0, 1.0, 0.0)
        self.up = Vector(0.0, 0.0, 1.0)

        self._name = name
        self._pen = _UP_
        self._orientation = _ORIENTATION_INIT # will be set to 1 or 2 later when used for residue building
        self._recording = False
        self._tape = []
    
    def copy_coords(self, source):
        '''
        Copy the Position, Heading, Left and Up coordinate system from the input source into self.
        Argument: source: Turtle3D
        Returns: None
        '''

        # copy the Arrays
        self._position = source._position.copy()
        self._heading = source._heading.copy()
        self._left = source._left.copy()
        self._up = source._up.copy()

        # copy the Vectors - create new ones from the source arrays
        self.position = Vector(source._position)
        self.heading = Vector(source._heading)
        self.left = Vector(source._left)
        self.up = Vector(source._up)

        self._orientation = source._orientation

    def reset(self):
        """
        Reset the Turtle to be at the Origin, with correct Heading, Left and Up vectors.
        Arguments: None
        Returns: None
        """
        self._position = numpy.array((0.0, 0.0, 0.0), "d")
        self._heading = numpy.array((1.0, 0.0, 0.0), "d")
        self._left = numpy.array((0.0, 1.0, 0.0), "d")
        self._up = numpy.array((0.0, 0.0, 1.0), "d")

        # expose these as Vectors
        self.position = Vector(0.0, 0.0, 0.0)
        self.heading = Vector(1.0, 0.0, 0.0)
        self.left = Vector(0.0, 1.0, 0.0)
        self.up = Vector(0.0, 0.0, 1.0)

        self._pen = _UP_
        self._orientation = -1
        self._recording = False
        self._tape = []
    
    def Orientation(self):
        return self._orientation
    
    def set_orientation(self, orientation):
        assert orientation == ORIENT_BACKBONE or orientation == ORIENT_SIDECHAIN, f'Orientation must be {ORIENT_BACKBONE} or {ORIENT_SIDECHAIN}'
        self._orientation = orientation
    
    def Pen(self):
        if self._pen == _UP_:
            return('UP')
        else:
            return('Down')

    def PenUp(self):
        self._pen = _UP_

    def PenDown(self):
        self._pen = _DOWN_

    def Recording(self):
        return self._recording

    def RecordOn(self):
        self._recording = True

    def RecordOff(self):
        self._recording = False

    def ResetTape(self):
        self._tape = []

    def setPosition(self, x, y=None, z=None):
        """
        Set the Turtle's Position.
            X is either a list or tuple.
        """
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._position = numpy.array(x, "d")
        else:
            # Three numbers
            self._position = numpy.array((x, y, z), "d")

        self.position = Vector(self._position)
        return

    def getPosition(self) -> numpy.array:
        """
        Get the Turtle's Position.
        Return: Turtle's position (Array)
        """
        return(self._position)

    def getVPosition(self) -> Vector:
        """
        Get the Turtle's Position.
        Return: Turtle's position (Array)
        """
        return(self._position)
    
    def getName(self):
        """
        Get the Turtle's Position.
        """
        return(self._name)
    
    def setName(self, name):
        """
        Set the Turtle'Name.
        
        """
        self._name = name
    
    def move(self, distance):
        """
        Move the Turtle distance, in direction of Heading
        """
        self._position = self._position + self._heading * distance

    def roll(self, angle):
        """
        Roll the Turtle about the heading vector angle degrees
        """
        
        ang = angle * math.pi / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        self._up[0] = cosang * self._up[0] - sinang * self._left[0]
        self._up[1] = cosang * self._up[1] - sinang * self._left[1]
        self._up[2] = cosang * self._up[2]- sinang * self._left[2]

        self._left[0] = cosang * self._left[0] + sinang * self._up[0]
        self._left[1] = cosang * self._left[1] + sinang * self._up[1]
        self._left[2] = cosang * self._left[2] + sinang * self._up[2]

    def yaw(self, angle):
        """
        Yaw the Turtle about the up vector (180 - angle) degrees. This is used when building molecules
        """
        
        ang = ((180 - angle) * math.pi) / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        self._heading[0] = cosang * self._heading[0] + sinang * self._left[0]
        self._heading[1] = cosang * self._heading[1] + sinang * self._left[1]
        self._heading[2] = cosang * self._heading[2] + sinang * self._left[2]

        self._left[0] = cosang * self._left[0] - sinang * self._heading[0]
        self._left[1] = cosang * self._left[1] - sinang * self._heading[1]
        self._left[2] = cosang * self._left[2] - sinang * self._heading[2]

    def turn(self, angle):
        """
        Turn the Turtle about the up vector angle degrees.
        """
        
        ang = (angle * math.pi) / 180.0

        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        self._heading[0] = cosang * self._heading[0] + sinang * self._left[0]
        self._heading[1] = cosang * self._heading[1] + sinang * self._left[1]
        self._heading[2] = cosang * self._heading[2] + sinang * self._left[2]

        self._left[0] = cosang * self._left[0] - sinang * self._heading[0]
        self._left[1] = cosang * self._left[1] - sinang * self._heading[1]
        self._left[2] = cosang * self._left[2] - sinang * self._heading[2]

    def pitch(self, angle):
        """
        pitch the Turtle about the left vector angle degrees
        """
        
        ang = angle * math.pi / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        self._heading[0] = self._heading[0] * cosang - self._up[0] * sinang
        self._heading[1] = self._heading[1] * cosang - self._up[1] * sinang
        self._heading[2] = self._heading[2] * cosang - self._up[2] * sinang

        self._up[0] = self._up[0] * cosang + self._heading[0] * sinang
        self._up[1] = self._up[1] * cosang + self._heading[1] * sinang
        self._up[2] = self._up[2] * cosang + self._heading[2] * sinang

    def _setHeading(self, x, y=None, z=None):
        """Set the Turtle's Heading.
            x is either a list or tuple.
        """
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._heading = numpy.array(x, "d")
        else:
            # Three numbers
            self._heading = numpy.array((x, y, z), "d")
        self.heading = Vector(self.heading)
        return
    
    def _setLeft(self, x, y=None, z=None):
        """Set the Turtle's Left.
            x is either a list or tuple.
        """
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._left = numpy.array(x, "d")
        else:
            # Three numbers
            self._left = numpy.array((x, y, z), "d")
        self.left = Vector(self._left)
        return
    
    def _setUp(self, x, y=None, z=None):
        """Set the Turtle's Up.
            x is either a list or tuple.
        """
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._up = numpy.array(x, "d")
        else:
            # Three numbers
            self._up = numpy.array((x, y, z), "d")
        self.up = Vector(self.up)
        return

    def unit(self, v):
        norm = numpy.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def orient(self, position: numpy.array, heading: numpy.array, left: numpy.array):
        """
        Orients the turtle with Position at p1, Heading at p2 and Left at p3
        Arguments:
            position
        """

        self._position = position

        temp = heading - position
        self._heading = self.unit(temp)
        self.heading = Vector(self._heading)

        temp = left - position
        self._left = self.unit(temp)

        temp = numpy.cross(self._heading, self._left)
        self._up = self.unit(temp)
        self.up = Vector(self._up)
        
        # fix left to be orthogonal
        temp = numpy.cross(self._up, self._heading)
        self._left = self.unit(temp)
        self.left = Vector(self._left)
        return

    def orient_at_residue(self, chain, resnumb, orientation):
        '''
        Orient the turtle at the specified residue from the input Chain in
        either orientation 1 or 2.

        Arguments: 
            turtle: input Turtle3D
            chain: list of Residues in the model, eg: chain = model['A']
            resnumb: residue number
            orientation: 
            1 - at Ca heading towards Cb with N at the left
            2 - at Ca heading towards C with N at the left
        Returns: None. Turtle internal state is modified
        '''

        assert self._orientation == 1 or self._orientation == 2, f'orient_at_residue() requires Turtle3D to be #1 or #2'
        
        residue = chain[resnumb]
        assert residue is not None, f'get_backbone_from_sidechain() requires valid residue number'

        # by this point I'm pretty confident I have coordinates
        # we pull the actual numpy.array from the coordinates since that's what the
        # Turtle3D expects

        n = residue['N'].get_vector().get_array()
        ca = residue['CA'].get_vector().get_array()
        cb = residue['CB'].get_vector().get_array()
        c = residue['C'].get_vector().get_array()

        if orientation == ORIENT_SIDECHAIN:
            self.orient(ca, cb, n)
            self.set_orientation(ORIENT_SIDECHAIN)
        elif orientation == ORIENT_BACKBONE:
            self.orient(ca, c, n)
            self.set_orientation(ORIENT_BACKBONE)
        return

    def orient_from_backbone(self, n: numpy.array, ca: numpy.array, cb: numpy.array, c: numpy.array, orientation):
        '''
        Orient the turtle at the specified residue from the input Chain in
        either orientation 1 or 2.

        Arguments: 
            turtle: input Turtle3D object
            n: position of n atom
            ca: position of ca atom
            c: position of c atom
            orientation: 
            1 - at Ca heading towards Cb with N at the left
            2 - at Ca heading towards C with N at the left
        Returns: None. Turtle internal state is modified
        '''

        assert orientation == 1 or orientation == 2, f'orient_at_residue() requires Turtle3D to be #1 or #2'

        
        _n = n.copy()
        _ca = ca.copy()
        _cb = cb.copy()
        _c = c.copy()
        
        if orientation == ORIENT_SIDECHAIN:
            self.orient(_ca, _cb, _n)
            self.set_orientation(ORIENT_SIDECHAIN)
        elif orientation == ORIENT_BACKBONE:
            self.orient(_ca, _c, _n)
            self.set_orientation(ORIENT_BACKBONE)
        return

    def to_local(self, global_vec) -> numpy.array:
        """
        Returns the Turtle-centered local coordinates for input Global vector (3d)
        """

        newpos = global_vec - self._position
        dp1 = numpy.dot(self._heading, newpos)
        dp2 = numpy.dot(self._left, newpos)
        dp3 = numpy.dot(self._up, newpos)

        result = numpy.array((dp1, dp2, dp3), "d")
        return result

    def to_localVec(self, global_vec) -> Vector:
        """
        Returns the Turtle-centered local coordinates for input Global vector (3d)
        """

        newpos = global_vec - self._position
        dp1 = numpy.dot(self._heading, newpos)
        dp2 = numpy.dot(self._left, newpos)
        dp3 = numpy.dot(self._up, newpos)

        return Vector(dp1, dp2, dp3)

    def to_global(self, local) -> numpy.array:
        """
        Returns the global coordinates for input local vector (3d)
        """

        p1 = self._position[0] + self._heading[0] * local[0] + self._left[0] * local[1] + self._up[0] * local[2]
        p2 = self._position[1] + self._heading[1] * local[0] + self._left[1] * local[1] + self._up[1] * local[2]
        p3 = self._position[2] + self._heading[2] * local[0] + self._left[2] * local[1] * self._up[2] * local[2]

        return numpy.array((p1, p2, p3), "d")

    def to_globalVec(self, local) -> Vector:
        """
        Returns the global coordinates for input local vector (3d)
        """

        p1 = self._position[0] + self._heading[0] * local[0] + self._left[0] * local[1] + self._up[0] * local[2]
        p2 = self._position[1] + self._heading[1] * local[0] + self._left[1] * local[1] + self._up[1] * local[2]
        p3 = self._position[2] + self._heading[2] * local[0] + self._left[2] * local[1] * self._up[2] * local[2]

        return Vector(p1, p2, p3)
    
    def __repr__(self):
        """Return Turtle 3D coordinates."""
        return f"<Turtle: {self._name}\n Position: {self._position},\n Heading: {self._heading} \n Left: {self._left} \n Up: {self._up}\n Orientation: {self._orientation}\n Pen: {self.Pen()} \n Recording: {self._recording}>"

    def bbone_to_schain(self):
        '''
        Function requires turtle to be in orientation #2 (at alpha carbon,
        headed towards carbonyl, with nitrogen on left) and converts to orientation #1
        (at alpha c, headed to beta carbon, with nitrogen on left.

        Arguments: 
            turtle: Turtle3D object in orientation #2

        Returns: modified Turtle3D
        '''

        assert self._orientation == 2, f'bbone_to_schain() requires Turtle3D to be in orientation #2'

        self.roll(240.0)
        self.pitch(180.0)
        self.yaw(110.0)
        self.roll(240.0)
        self.set_orientation(1) # sets the orientation flag


    def schain_to_bbone(self):
        '''
        Function requires turtle to be in orientation #1 (at alpha c, headed to beta carbon, with nitrogen on left)
        and converts to orientation #2 (at alpha carbon, headed towards carbonyl, with nitrogen on left).

        Arguments: 
            None
        Returns: modified Turtle3D
        '''

        assert self._orientation == 1, f'schain_to_bbone() requires Turtle3D to be in orientation #1'

        self.pitch(180.0)
        self.roll(240.0)
        self.yaw(110.0)
        self.roll(120.0)
        self.set_orientation(2) # sets the orientation flag
        return

# End of file
