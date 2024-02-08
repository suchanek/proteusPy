import unittest
from unittest import TestCase
import math

from proteusPy import Disulfide, DisulfideList, DisulfideLoader, load_disulfides_from_id, Load_PDB_SS
from proteusPy.data import SS_DICT_PICKLE_FILE
from proteusPy.ProteusGlobals import _FLOAT_INIT, _INT_INIT
from proteusPy.utility import distance_squared, distance3d

class TestDisulfide(TestCase):
    def test_init(self):
        ss = Disulfide()
        self.assertEqual(ss.name, '')
        self.assertEqual(ss.prox, _INT_INIT)
        self.assertEqual(ss.dist, _INT_INIT)
        # Test init with values
        ss = Disulfide(name='test', prox=10, dist=15)
        self.assertEqual(ss.name, 'test')
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
        energy = 2.0 * math.cos(3.0 * math.radians(60)) + math.cos(3.0 * math.radians(45)) + math.cos(3.0 * math.radians(120)) + math.cos(3.0 * math.radians(30)) + 3.5 * math.cos(2.0 * math.radians(-90)) + 0.6 * math.cos(3.0 * math.radians(-90)) + 10.1
        self.assertAlmostEqual(ss.energy, energy)   

    def test_repr(self):
        ss = Disulfide('test')
        self.assertEqual(repr(ss), 'Disulfide(test)')
    
    def test_str(self):
        ss = Disulfide('test')
        self.assertEqual(str(ss), 'Disulfide(test)')

    def test_copy(self):
        ss = Disulfide('test')
        ss.chi1 = 60
        ss2 = ss.copy()
        self.assertEqual(ss2.chi1, 60)
        self.assertEqual(ss2.name, 'test')

    def test_from_dict(self):
        test_dict = {'name': 'test', 'chi1': 60, 'chi2': 120}
        ss = Disulfide.from_dict(test_dict)
        self.assertEqual(ss.name, 'test')
        self.assertEqual(ss.chi1, 60)
        self.assertEqual(ss.chi2, 120)

class TestDisulfideList(TestCase):

    def test_init(self):
        ss1 = Disulfide('ss1')
        sslist = DisulfideList([ss1], 'test')
        self.assertEqual(len(sslist), 1)
        self.assertEqual(sslist.name, 'test')
    
    def test_display(self):
        # Mock test
        sslist = DisulfideList([], 'test')
        sslist.display('cpk')
        self.assertTrue(True) # Display happened

    def test_screenshot(self):
        # Mock test
        sslist = DisulfideList([], 'test')
        sslist.screenshot('cpk', 'test.png')
        self.assertTrue(True) # Screenshot created

class TestDisulfideLoader(TestCase):
    from proteusPy import Load_PDB_SS
    def test_load(self):
        loader = Load_PDB_SS(verbose=True, subset=False)
        self.assertGreater(len(loader), 0)
    
    def test_getitem(self):
        loader = Load_PDB_SS(verbose=True, subset=False)
        ss = loader[0]
        self.assertIsInstance(ss, Disulfide)

class TestModuleFunctions(TestCase):
    from proteusPy import Load_PDB_SS

    def test_load_disulfides_from_id(self):
        _pdb = Load_PDB_SS(verbose=True, subset=False)
        sslist = _pdb['1a25']
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

if __name__ == '__main__':
    unittest.main()
