import unittest

import numpy as np
from numpy.testing import assert_array_equal

from proteusPy.turtle3D import Turtle3D
from proteusPy.vector3D import Vector3D

# Define a _tolerance threshold
_tolerance = 1e-8


def cmp_vec(v1: Vector3D, v2: Vector3D, tol: float) -> bool:
    "Return true if the length of the difference between the two vectors is less than a tolerance."
    _diff = v2 - v1
    _len = _diff.magnitude()
    return _len < tol


class TestTurtle3D(unittest.TestCase):

    def test_init(self):
        """
        Test initializing a Turtle3D object.

        Verifies that the position, heading, left and up vectors are
        initialized to the expected default values.
        """

        turtle = Turtle3D()
        assert_array_equal(turtle._position, np.array((0.0, 0.0, 0.0)))
        assert_array_equal(turtle._heading, np.array((1.0, 0.0, 0.0)))
        assert_array_equal(turtle._left, np.array((0.0, 1.0, 0.0)))
        assert_array_equal(turtle._up, np.array((0.0, 0.0, 1.0)))

    def test_forward(self):
        """
        Test moving the Turtle3D object.

        Verifies that the position vector is the expected value after moving forward.
        """
        _pos = Vector3D(10.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.move(10.0)
        assert cmp_vec(turtle.pos, _pos, _tolerance)

    def test_back(self):
        """
        Test moving the Turtle3D object.

        Verifies that the position vector is the expected value after moving backward.
        """
        _pos = Vector3D(5.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.move(10.0)
        turtle.move(-5.0)
        assert cmp_vec(turtle.pos, _pos, _tolerance)

    def test_right(self):
        """
        Test turning the Turtle3D object.

        Verifies that the heading vector is the expected value after turning right.
        """
        _head = Vector3D(0.0, 1.0, 0.0)
        turtle = Turtle3D()
        turtle.turn(90.0)
        _h2 = turtle.h
        assert cmp_vec(_h2, _head, _tolerance)

    def test_left(self):
        """
        Test turning the Turtle3D object.

        Verifies that the heading vector is the expected value after turning left.
        """

        _head = Vector3D(0.0, -1.0, 0.0)
        turtle = Turtle3D()
        turtle.turn(-90.0)
        assert cmp_vec(turtle.h, _head, _tolerance)

    def test_roll(self):
        """
        Test rolling the Turtle3D object.

        Verifies that the left and up vectors are the expected values after rolling.
        """

        _left = Vector3D(0.0, 0.0, 1.0)
        _up = Vector3D(0.0, -1.0, 0.0)
        turtle = Turtle3D()
        turtle.roll(90)
        assert cmp_vec(turtle.l, _left, _tolerance)
        assert cmp_vec(turtle.u, _up, _tolerance)

    def test_pitch(self):
        """
        Test pitching the Turtle3D object.

        Verifies that the heading and up vectors are the expected values after pitching up.
        """

        _head = Vector3D(0.0, 0.0, -1.0)
        _up = Vector3D(1.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.pitch(90)
        assert cmp_vec(turtle.h, _head, _tolerance)
        assert cmp_vec(turtle.u, _up, _tolerance)

    def test_yaw(self):
        """
        Test yawing the Turtle3D object.

        Verifies that the heading and left vectors are the expected values after yawing.
        """

        _head = Vector3D(0.0, 1.0, 0.0)
        _left = Vector3D(-1.0, 0.0, 0.0)

        turtle = Turtle3D()
        turtle.yaw(90)
        assert cmp_vec(turtle.h, _head, _tolerance)
        assert cmp_vec(turtle.l, _left, _tolerance)

    def test_new(self):
        _pos = Vector3D(100.0, 0.0, 0.0)
        _head = Vector3D(10.0, 0.0, 0.0)
        turtle = Turtle3D()
        turtle.new("test", _pos, _head)
        assert turtle.Name == "test"
        assert cmp_vec(turtle.Position, _pos, _tolerance)
        assert cmp_vec(turtle.Heading, _head.normalized(), _tolerance)
