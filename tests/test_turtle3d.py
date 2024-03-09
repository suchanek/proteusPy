import unittest

import numpy as np
from Bio.PDB.vectors import Vector
from numpy.testing import assert_array_equal

from proteusPy.turtle3D import Turtle3D

# Define a _tolerance threshold
_tolerance = 1e-8


def cmp_vec(v1: Vector, v2: Vector, tol: float) -> bool:
    "Return true if the length of the difference between the two vectors is less than a tolerance."
    _diff = v2 - v1
    _len = _diff.norm()
    return _len < tol


class TestTurtle3D(unittest.TestCase):

    def test_init(self):
        turtle = Turtle3D()
        assert_array_equal(turtle._position, np.array((0.0, 0.0, 0.0)))
        assert_array_equal(turtle._heading, np.array((1.0, 0.0, 0.0)))
        assert_array_equal(turtle._left, np.array((0.0, 1.0, 0.0)))
        assert_array_equal(turtle._up, np.array((0.0, 0.0, 1.0)))
    
    def test_forward(self):
        _pos = Vector(10.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.move(10.0)
        assert cmp_vec(turtle.position, _pos, _tolerance)

    def test_back(self):
        _pos = Vector(5.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.move(10.0)
        turtle.move(-5.0)
        assert cmp_vec(turtle.position, _pos, _tolerance)

    def test_right(self):
        _head = Vector(0.0, 1.0, 0.0)
        turtle = Turtle3D()
        turtle.turn(90.0)
        _h2 = turtle.heading
        assert cmp_vec(_h2, _head, _tolerance)

    def test_left(self):
        _head = Vector(0.0, -1.0, 0.0)
        turtle = Turtle3D()
        turtle.turn(-90.0)
        assert cmp_vec(turtle.heading, _head, _tolerance)

    def test_roll(self):
        _left = Vector(0.0, 0.0, 1.0)
        _up = Vector(0.0, -1.0, 0.0)
        turtle = Turtle3D()
        turtle.roll(90)
        assert cmp_vec(turtle.left, _left, _tolerance)
        assert cmp_vec(turtle.up, _up, _tolerance)

    def test_pitch(self):
        _head = Vector(0.0, 0.0, -1.0)
        _up = Vector(1.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.pitch(90)
        assert cmp_vec(turtle.heading, _head, _tolerance)

    def test_yaw(self):
        _head = Vector(0.0, 1.0, 0.0)
        _left = Vector(-1.0, 0.0, 0.0)

        turtle = Turtle3D()
        turtle.yaw(90)
        assert cmp_vec(turtle.heading, _head, _tolerance)
