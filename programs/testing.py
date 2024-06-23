import math
import unittest
from unittest import TestCase

from proteusPy import Disulfide, DisulfideList, Load_PDB_SS
from proteusPy.data import SS_DICT_PICKLE_FILE
from proteusPy.ProteusGlobals import _FLOAT_INIT, _INT_INIT
from proteusPy.utility import distance3d, distance_squared


class TestDisulfide(TestCase):
    from proteusPy.Disulfide import Disulfide

    def test_init(self):
        ss = Disulfide()
        self.assertEqual(ss.name, "SSBOND")
        self.assertEqual(ss.proximal, _INT_INIT)
        self.assertEqual(ss.distal, _INT_INIT)
        # Test init with values
        ss = Disulfide(name="test", proximal=10, distal=15)
        self.assertEqual(ss.name, "test")
        self.assertEqual(ss.proximal, 10)
        self.assertEqual(ss.distal, 15)

    def test_energy(self):
        from proteusPy import Disulfide_Energy_Function

        ss = Disulfide()
        # Test with sample torsions
        ss.dihedrals = [60, 120, -90, 30, 45]
        energy = Disulfide_Energy_Function(ss.dihedrals)
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
        ss.build_model(-60.0, -60.0, -90.0, -60.0, -60.0)
        ss.display("cpk")
        self.assertTrue(True)  # Display happened


PDB_SS = Load_PDB_SS(verbose=True, subset=True)


class TestDisulfideList(TestCase):
    from proteusPy.DisulfideList import DisulfideList

    def test_init(self):
        global PDB_SS

        # PDB_SS = Load_PDB_SS(verbose=True, subset=True)
        sslist = PDB_SS[0:10]
        self.assertEqual(len(sslist), 10)
        self.assertEqual(sslist.pdb_id, "4yys")

    def test_display(self):
        # Mock test
        global PDB_SS

        # PDB_SS = Load_PDB_SS(verbose=True, subset=True)
        sslist = PDB_SS[0:10]

        sslist = DisulfideList(PDB_SS[0:10], "4yys")
        sslist.display("cpk")
        self.assertTrue(True)  # Display happened

    def test_screenshot(self):
        # Mock test
        global PDB_SS
        # PDB_SS = Load_PDB_SS(verbose=True, subset=True)
        sslist = PDB_SS[0:10]
        sslist.display(style="cpk", panelsize=256)
        self.assertTrue(True)  # Screenshot created


class TestDisulfideLoader(TestCase):
    from proteusPy import Load_PDB_SS

    # PDB_SS = Load_PDB_SS(verbose=True, subset=True)

    def test_load(self):
        global PDB_SS
        # loader = Load_PDB_SS(verbose=True, subset=True)
        self.assertIsNotNone(PDB_SS)

    def test_getitem(self):
        global PDB_SS

        # loader = Load_PDB_SS(verbose=True, subset=True)
        ss = PDB_SS[0]
        self.assertIsInstance(ss, Disulfide)


class TestModuleFunctions(TestCase):
    from proteusPy import Load_PDB_SS

    def test_load_disulfides_from_id(self):
        global PDB_SS

        # _pdb = Load_PDB_SS(verbose=True, subset=False)
        sslist = PDB_SS["1a25"]
        self.assertIsNone(sslist)

    def test_distance_squared(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        dist2 = distance_squared(v1, v2)
        self.assertEqual(dist2, 2)

    def test_distance3d(self):
        from proteusPy import Vector

        v1 = Vector(1, 0, 0)
        v2 = Vector(0, 1, 0)
        dist = distance3d(v1, v2)
        self.assertEqual(dist, 2**0.5)


if __name__ == "__main__":
    unittest.main()
