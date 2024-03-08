import unittest

import numpy
from Bio.PDB.vectors import Vector
from numpy.testing import assert_array_equal

from proteusPy.turtle3D import Turtle3D


class TestTurtle3D(unittest.TestCase):

    def test_init(self):
        turtle = Turtle3D()
        assert_array_equal(turtle._position, numpy.array((0.0, 0.0, 0.0)))
        assert_array_equal(turtle._heading, numpy.array((1.0, 0.0, 0.0)))
        assert_array_equal(turtle._left, numpy.array((0.0, 1.0, 0.0)))
        assert_array_equal(turtle._up, numpy.array((0.0, 0.0, 1.0)))

    def test_new(self):
        turtle = Turtle3D()
        turtle.new("test", Vector((1.0, 2.0, 3.0)), Vector((4.0, 5.0, 6.0)))
        assert turtle._name == "test"
        assert_array_equal(turtle._position, numpy.array((1.0, 2.0, 3.0)))
        assert_array_equal(turtle._heading, numpy.array((4.0, 5.0, 6.0)))

    def test_forward(self):
        turtle = Turtle3D()
        turtle.move(10)
        assert_array_equal(turtle._position, numpy.array((10.0, 0.0, 0.0)))

    def test_back(self):
        turtle = Turtle3D()
        turtle.move(10)
        turtle.move(-5)
        assert_array_equal(turtle._position, numpy.array((5.0, 0.0, 0.0)))

    def test_right(self):
        turtle = Turtle3D()
        turtle.turn(90)
        assert_array_equal(turtle._heading, numpy.array((0.0, 1.0, 0.0)))

    def test_left(self):
        turtle = Turtle3D()
        turtle.turn(-90)
        assert_array_equal(turtle._heading, numpy.array((0.0, -1.0, 0.0)))

    def test_roll(self):
        turtle = Turtle3D()
        turtle.roll(90)
        assert_array_equal(turtle._left, numpy.array((0.0, 0.0, 1.0)))
        assert_array_equal(turtle._up, numpy.array((-1.0, 0.0, 0.0)))

    def test_pitch(self):
        turtle = Turtle3D()
        turtle.pitch(90)
        assert turtle._heading == numpy.array((0.0, 0.0, -1.0))
        assert turtle._up == numpy.array((0.0, 1.0, 0.0))

    def test_yaw(self):
        turtle = Turtle3D()
        turtle.yaw(90)
        assert turtle._heading == numpy.array((0.0, 1.0, 0.0))
        assert turtle._left == numpy.array((0.0, 0.0, 1.0))
