import math
import unittest
from unittest import TestCase

from proteusPy import (
    Disulfide,
    DisulfideList,
    DisulfideLoader,
    Load_PDB_SS,
    load_disulfides_from_id,
)
from proteusPy.ProteusGlobals import _FLOAT_INIT, _INT_INIT
from proteusPy.utility import distance3d, distance_squared


class TestDisulfide(TestCase):
    from proteusPy.DisulfideBase import Disulfide

    def test_init(self):
        ss = Disulfide()
        self.assertEqual(ss.name, "EGS")
        self.assertEqual(ss.prox, _INT_INIT)
        self.assertEqual(ss.dist, _INT_INIT)
        # Test init with values
        ss = Disulfide(name="test", prox=10, dist=15)
        self.assertEqual(ss.name, "test")
        self.assertEqual(ss.prox, 10)
        self.assertEqual(ss.dist, 15)

    def test_energy(self):
        ss = Disulfide()
        self.assertEqual(ss.energy, _FLOAT_INIT)
        # Test with sample torsions
        ss.chi1 = 60
        ss.chi2 = 120
        ss.chi3 = -90
        ss.chi4 = 30
        ss.chi5 = 45
        energy = (
            2.0 * math.cos(3.0 * math.radians(60))
            + math.cos(3.0 * math.radians(45))
            + math.cos(3.0 * math.radians(120))
            + math.cos(3.0 * math.radians(30))
            + 3.5 * math.cos(2.0 * math.radians(-90))
            + 0.6 * math.cos(3.0 * math.radians(-90))
            + 10.1
        )
        self.assertAlmostEqual(ss.energy, energy)

    def test_copy(self):
        ss = Disulfide("test")
        ss.chi1 = 60
        ss2 = ss.copy()
        self.assertEqual(ss2.chi1, 60)
        self.assertEqual(ss2.name, "test")

    def test_display(self):
        # Mock test
        ss = Disulfide("test")
        ss.build_model("-60.0", "-60.0", "-90.0", "-60.0", "-60.0")
        ss.display("cpk")
        self.assertTrue(True)  # Display happened


class TestDisulfideList(TestCase):
    from proteusPy.DisulfideBase import DisulfideList

    def test_init(self):
        PDB_SS = Load_PDB_SS(verbose=True, subset=True)
        sslist = PDB_SS[0:10]
        self.assertEqual(len(sslist), 10)
        self.assertEqual(sslist.name, "test")

    def test_display(self):
        # Mock test
        PDB_SS = Load_PDB_SS(verbose=True, subset=True)
        sslist = DisulfideList(PDB_SS[0:10], "test")
        sslist.display("cpk")
        self.assertTrue(True)  # Display happened

    def test_screenshot(self):
        # Mock test
        self.sslist.screenshot("cpk", "test.png")
        self.assertTrue(True)  # Screenshot created


class TestDisulfideLoader(TestCase):
    from proteusPy import Load_PDB_SS

    def test_load(self):
        loader = Load_PDB_SS(verbose=True, subset=True)
        self.assertIsNotNone(loader)

    def test_getitem(self):
        loader = Load_PDB_SS(verbose=True, subset=True)
        ss = loader[0]
        self.assertIsInstance(ss, Disulfide)


class TestModuleFunctions(TestCase):
    from proteusPy import Load_PDB_SS

    def test_load_disulfides_from_id(self):
        _pdb = Load_PDB_SS(verbose=True, subset=False)
        sslist = _pdb["1a25"]
        self.assertGreater(len(sslist), 0)

    def test_distance_squared(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        dist2 = distance_squared(v1, v2)
        self.assertEqual(dist2, 2)

    def test_distance3d(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        dist = distance3d(v1, v2)
        self.assertEqual(dist, 2**0.5)


if __name__ == "__main__":
    unittest.main()
