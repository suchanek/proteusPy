import unittest

from proteusPy.turtle3D import Turtle3D
from Bio.PDB.vectors import Vector


class TestTurtle3D(unittest.TestCase):

    def test_init(self):
        turtle = Turtle3D()
        self.assertEqual(turtle.position, Vector(0.0, 0.0, 0.0))
        self.assertEqual(turtle.heading, Vector(1.0, 0.0, 0.0))
        self.assertEqual(turtle.left, Vector(0.0, 1.0, 0.0))
        self.assertEqual(turtle.up, Vector(0.0, 0.0, 1.0))

    def test_new(self):
        turtle = Turtle3D.new("test", Vector(1, 2, 3), Vector(4, 5, 6))
        self.assertEqual(turtle.name, "test")
        self.assertEqual(turtle.position, Vector(1, 2, 3))
        self.assertEqual(turtle.heading, Vector(4, 5, 6))

    def test_forward(self):
        turtle = Turtle3D()
        turtle.forward(10)
        self.assertEqual(turtle.position, Vector(10, 0, 0))

    def test_back(self):
        turtle = Turtle3D()
        turtle.forward(10)
        turtle.back(5)
        self.assertEqual(turtle.position, Vector(5, 0, 0))

    def test_right(self):
        turtle = Turtle3D()
        turtle.right(90)
        self.assertEqual(turtle.heading, Vector(0, 1, 0))

    def test_left(self):
        turtle = Turtle3D()
        turtle.left(90)
        self.assertEqual(turtle.heading, Vector(0, -1, 0))

    def test_roll(self):
        turtle = Turtle3D()
        turtle.roll(90)
        self.assertEqual(turtle.left, Vector(0, 0, 1))
        self.assertEqual(turtle.up, Vector(-1, 0, 0))

    def test_pitch(self):
        turtle = Turtle3D()
        turtle.pitch(90)
        self.assertEqual(turtle.heading, Vector(0, 0, -1))
        self.assertEqual(turtle.left, Vector(0, 1, 0))

    def test_yaw(self):
        turtle = Turtle3D()
        turtle.yaw(90)
        self.assertEqual(turtle.heading, Vector(0, 1, 0))
        self.assertEqual(turtle.up, Vector(0, 0, 1))
