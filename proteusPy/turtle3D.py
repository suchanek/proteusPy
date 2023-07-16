'''
Implementation of a 3D 'Turtle' in Python.

Part of the program proteusPy, https://github.com/suchanek/proteusPy, 
a Python packages for the manipulation and analysis of macromolecules. 
Based on the C implementation originally authored by Eric G. Suchanek PhD, 1990.

'''

# Last modification 7/15/23 -egs-

__pdoc__ = {'__all__': True}

import numpy
import math

numpy.set_printoptions(suppress=True)

from Bio.PDB.vectors import Vector
from Bio.PDB.vectors import calc_angle, calc_dihedral

#from proteusPy import *
from proteusPy.ProteusGlobals import ORIENT_BACKBONE, ORIENT_SIDECHAIN

_DOWN_ = -1
_UP_ = 1
_ORIENTATION_INIT = -1

ORIENT_BACKBONE = 2
ORIENT_SIDECHAIN = 1

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
    
    def new(self, name: str, pos: Vector, 
            head=Vector(1,0,0), 
            left=Vector(0,1,0), 
            up=Vector(0,0,1), 
            pen='up', recording=False):
        '''
        Initialize a Turtle with a name and initial position

        :param name: Turtle's Name
        :param pos: Turtle's Position 
        :param head: Turtle's Heading vector, defaults to Vector(1,0,0)
        :param left: Turtle's Left vector, defaults to Vector(0,1,0)
        :param up: Turtle's Up vector, defaults to Vector(0,0,1)
        :param pen: Pen state, defaults to 'up'
        :param recording: _description_, defaults to False
        '''
        self.Name = name
        self.Position = pos
        self.Heading = head
        self.Left = left
        self.Up = up
        self.Pen = pen
        self.Recording = recording

    def copy_coords(self, source):
        '''
        Copy the Position, Heading, Left and Up coordinate system from 
        the input source into self. Argument: source: Turtle3D
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
        self.__init__()
    
    @property
    def Orientation(self):
        return self._orientation
    
    @Orientation.setter
    def Orientation(self, orientation):
        assert orientation == ORIENT_BACKBONE or orientation == ORIENT_SIDECHAIN, \
         f'Orientation must be {ORIENT_BACKBONE} or {ORIENT_SIDECHAIN}'
        self._orientation = orientation
    
    @property
    def Pen(self):
        if self._pen == _UP_:
            return('up')
        else:
            return('down')
    
    @Pen.setter
    def Pen(self, pen):
        if pen == 'up':
            self._pen = _UP_
        elif pen == 'down':
            self._pen = _DOWN_
        else:
            self._pen = _DOWN_
    
    @property
    def Recording(self):
        return self._recording

    @Recording.setter
    def Recording(self, recording: bool):
        self._recording = recording

    def ResetTape(self):
        self.Recording(False)
        self._tape = []

    @property 
    def Position(self):
        """
        The Turtle's Position

        :return: Position
        :rtype: Vector
        """
        return(self.position)

    @Position.setter
    def Position(self, x, y=None, z=None):
        """
        Set's the Turtle's Position

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate, defaults to None
        :type y: float, optional
        :param z: Z coordinate, defaults to None
        :type z: float, optional
        :raises ValueError: Type error
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
    
    @property 
    def Heading(self):
        """
        Get the Turtle's Heading

        :return: Heading
        :rtype: Vector
        """
        return(self.heading)

    @Heading.setter
    def Heading(self, x, y=None, z=None):
        """
        Set the turtle's Heading direction vector

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate, defaults to None
        :type y: float, optional
        :param z: Z coordinate, defaults to None
        :type z: float, optional
        :raises ValueError: illegal value
        """

        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._heading = numpy.array(x, "d")
        else:
            # Three numbers
            self._heading = numpy.array((x, y, z), "d")
        self.heading = Vector(self._heading)
        return
    
    @property 
    def Left(self):
        """
        Get the Turtle's Left direction vector

        :return: Left
        :rtype: Vector
        """

        return(self.left)

    @Left.setter
    def Left(self, x, y=None, z=None):
        """
        Set the turtle's Left direction vector

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate, defaults to None
        :type y: float, optional
        :param z: Z coordinate, defaults to None
        :type z: float, optional
        :raises ValueError: illegal value
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
    
    @property
    def Up(self):
        """
        The Turtle's Up direction vector

        :return: Up
        :rtype: Vector
        """

        return(self.up)

    @Up.setter
    def Up(self, x, y=None, z=None):
        """
        Set the turtle's Up direction vector

        :param x: X coordinate
        :type x: float
        :param y: Y coordinate, defaults to None
        :type y: float, optional
        :param z: Z coordinate, defaults to None
        :type z: float, optional
        :raises ValueError: illegal value
        """
        if y is None and z is None:
            # Array, list, tuple...
            if len(x) != 3:
                raise ValueError("Turtle3D: x is not a list/tuple/array of 3 numbers")
            self._up = numpy.array(x, "d")
        else:
            # Three numbers
            self._up = numpy.array((x, y, z), "d")
        self.up = Vector(self._up)
        return

    def VPosition(self) -> Vector:
        """
        Return the Turtle's Position as a Vector

        :return: Position
        :rtype: Vector
        """
        return(self._position)
    
    @property
    def Name(self):
        """
        Return the Turtle's Name.
        """
        return(self._name)
    
    @Name.setter
    def Name(self, name):
        """
        Set the Turtle's name.
        
        """
        self._name = name
    
    def move(self, distance: float):
        """
        Move the Turtle distance (Å), in direction of Heading

        :param distance: Amount to move (Å)
        :type distance: float
        """
        self._position = self._position + self._heading * distance
        self.position = Vector(self._position)

    def roll(self, angle):
        """
        Roll the Turtle about the heading vector angle degrees

        :param angle: roll angle, degrees -180 -> 180
        :type angle: float
        """
        
        ang = angle * math.pi / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        lold = self._left.copy()
        uold = self._up.copy()

        self._up[0] = cosang * uold[0] - sinang * lold[0]
        self._up[1] = cosang * uold[1] - sinang * lold[1]
        self._up[2] = cosang * uold[2] - sinang * lold[2]
        self._up = self.unit(self._up)

        self.up = Vector(self._up)

        self._left[0] = cosang * lold[0] + sinang * uold[0]
        self._left[1] = cosang * lold[1] + sinang * uold[1]
        self._left[2] = cosang * lold[2] + sinang * uold[2]
        self._left = self.unit(self._left)
        
        self.left = Vector(self._left)

    def yaw(self, angle):
        """
        Yaw the Turtle about the up vector (180 - angle) degrees. 
        This is used when building molecules

        :param angle: Yaw angle, degrees -180 -> 180
        :type angle: float
        """
        
        ang = ((180 - angle) * math.pi) / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        lold = self._left.copy()
        hold = self._heading.copy()
        
        self._heading[0] = cosang * hold[0] + sinang * lold[0]
        self._heading[1] = cosang * hold[1] + sinang * lold[1]
        self._heading[2] = cosang * hold[2] + sinang * lold[2]
        self._heading = self.unit(self._heading)
        self.heading = Vector(self._heading)

        self._left[0] = cosang * lold[0] - sinang * hold[0]
        self._left[1] = cosang * lold[1] - sinang * hold[1]
        self._left[2] = cosang * lold[2] - sinang * hold[2]
        self._left = self.unit(self._left)
        self.left = Vector(self._left)

    def turn(self, angle):
        """
        Turn the Turtle about the up vector angle degrees.

        :param angle: Turn angle, degrees
        :type angle: float
        """
        
        ang = (angle * math.pi) / 180.0

        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        heading = self._heading.copy()
        left = self._left.copy()

        self._heading[0] = cosang * heading[0] + sinang * left[0]
        self._heading[1] = cosang * heading[1] + sinang * left[1]
        self._heading[2] = cosang * heading[2] + sinang * left[2]

        self._heading = self.unit(self._heading)
        self.heading = Vector(self._heading)

        self._left[0] = cosang * left[0] - sinang * heading[0]
        self._left[1] = cosang * left[1] - sinang * heading[1]
        self._left[2] = cosang * left[2] - sinang * heading[2]
        self._left = self.unit(self._left)
        self.left = Vector(self._left)

    def pitch(self, angle):
        """
        pitch the Turtle about the left vector angle degrees

        :param angle: Pitch angle, degrees -180 -> 180
        :type angle: float
        """
        
        up = self._up.copy()
        heading = self._heading.copy()

        ang = angle * math.pi / 180.0
        cosang = numpy.cos(ang)
        sinang = numpy.sin(ang)

        self._heading[0] = heading[0] * cosang - up[0] * sinang
        self._heading[1] = heading[1] * cosang - up[1] * sinang
        self._heading[2] = heading[2] * cosang - up[2] * sinang
        self._heading = self.unit(self._heading)
        self.heading = Vector(self._heading)

        self._up[0] = up[0] * cosang + heading[0] * sinang
        self._up[1] = up[1] * cosang + heading[1] * sinang
        self._up[2] = up[2] * cosang + heading[2] * sinang
        self._up = self.unit(self._up)
        self.up = Vector(self._up)
    
    def unit(self, v):
        '''
        Return a unit vector for the input vector.

        :param v: Input Vector
        :return: Unit Vector
        '''
        norm = numpy.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def orient(self, position: numpy.array, heading: numpy.array, left: numpy.array):
        """
        Orients the turtle with Position at p1, Heading at p2 and Left at p3

        :param position: Position
        :type position: numpy.array
        :param heading: Heading direction vector
        :type heading: numpy.array
        :param left: Left direction vector
        :type left: numpy.array
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
        """
        Orient the turtle at the specified residue from the input Chain in
        either orientation 1 or 2.

        :param chain: list of Residues in the model, eg: chain = model['A']
        :type chain: str
        :param resnumb: residue number
        :type resnumb: int
        :param orientation: 1 - at Ca heading towards Cb with N at the left or
            2 - at Ca heading towards C with N at the left
        :type orientation: int
        """
        
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
            self.Orientation = ORIENT_SIDECHAIN
        elif orientation == ORIENT_BACKBONE:
            self.orient(_ca, _c, _n)
            self.Orientation = ORIENT_BACKBONE
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

        res = self.to_local(global_vec)
        return Vector(res)

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
        res = self.to_global(local)
        return Vector(res)
    
    def __repr__(self):
        """Return Turtle 3D coordinates."""
        return f"<Turtle: {self._name}\n Position: {self._position},\n Heading: {self._heading} \n Left: {self._left} \n Up: {self._up}\n Orientation: {self._orientation}\n Pen: {self._pen} \n Recording: {self._recording}>"

    def bbone_to_schain(self):
        '''
        Function requires turtle to be in orientation #2 (at alpha carbon,
        headed towards carbonyl, with nitrogen on left) and converts to orientation #1
        (at alpha c, headed to beta carbon, with nitrogen on left.

        Arguments: 
            turtle: Turtle3D object in orientation #2

        Returns: modified Turtle3D
        '''

        # assert self._orientation == 2, f'bbone_to_schain() requires Turtle3D to be in orientation #2'

        self.roll(240.0)
        self.pitch(180.0)
        self.yaw(110.0)
        self.roll(240.0)
        self.Orientation = 1 # sets the orientation flag

    def schain_to_bbone(self):
        '''
        Function requires turtle to be in orientation #1 (at alpha c, headed to beta carbon, with nitrogen on left)
        and converts to orientation #2 (at alpha carbon, headed towards carbonyl, with nitrogen on left).

        Arguments: 
            None
        Returns: modified Turtle3D
        '''

        # assert self._orientation == 1, f'schain_to_bbone() requires Turtle3D to be in orientation #1'

        self.pitch(180.0)
        self.roll(240.0)
        self.yaw(110.0)
        self.roll(120.0)
        self.Orientation = 2 # sets the orientation flag
        return

# class definition ends

def test_turtle():
    '''
    Tests the turtle move, roll, yaw, pitch and turn.

    >>> import proteusPy
    >>> from proteusPy.turtle3D import Turtle3D
    >>> turt = Turtle3D('test')
    >>> turt.move(2.0)
    >>> turt
    <Turtle: test
     Position: [2. 0. 0.],
     Heading: [1. 0. 0.] 
     Left: [0. 1. 0.] 
     Up: [0. 0. 1.]
     Orientation: -1
     Pen: 1 
     Recording: False>
    >>> turt.roll(45.0)
    >>> turt
    <Turtle: test
     Position: [2. 0. 0.],
     Heading: [1. 0. 0.] 
     Left: [0.         0.70710678 0.70710678] 
     Up: [ 0.         -0.70710678  0.70710678]
     Orientation: -1
     Pen: 1 
     Recording: False>
    >>> turt.roll(45.0)
    >>> turt
    <Turtle: test
     Position: [2. 0. 0.],
     Heading: [1. 0. 0.] 
     Left: [0. 0. 1.] 
     Up: [ 0. -1.  0.]
     Orientation: -1
     Pen: 1 
     Recording: False>
    >>> turt.reset()
    >>> turt.pitch(45.0)
    >>> turt
    <Turtle: 3D_Turtle
     Position: [0. 0. 0.],
     Heading: [ 0.70710678  0.         -0.70710678] 
     Left: [0. 1. 0.] 
     Up: [0.70710678 0.         0.70710678]
     Orientation: -1
     Pen: 1 
     Recording: False>
    >>> turt.pitch(45.0)
    >>> turt
    <Turtle: 3D_Turtle
     Position: [0. 0. 0.],
     Heading: [ 0.  0. -1.] 
     Left: [0. 1. 0.] 
     Up: [1. 0. 0.]
     Orientation: -1
     Pen: 1 
     Recording: False>
>>> turt.yaw(45.0)
    >>> turt
    <Turtle: 3D_Turtle
     Position: [0. 0. 0.],
     Heading: [-0.          0.70710678  0.70710678] 
     Left: [-0.         -0.70710678  0.70710678] 
     Up: [1. 0. 0.]
     Orientation: -1
     Pen: 1 
     Recording: False>
    >>> turt.yaw(45.0)
    >>> turt
    <Turtle: 3D_Turtle
     Position: [0. 0. 0.],
     Heading: [-0. -1.  0.] 
     Left: [ 0. -0. -1.] 
     Up: [1. 0. 0.]
     Orientation: -1
     Pen: 1 
     Recording: False>

    '''
    return

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# End of file
