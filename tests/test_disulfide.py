"""
Unit tests for the Disulfide class in the proteusPy package.

This module contains a set of unit tests for verifying the functionality of the Disulfide class
and related functions. The tests are implemented using the unittest framework.

Classes:
    TestDisulfide: Contains unit tests for the Disulfide class.

Methods:
    setUp: Initializes the test environment, loads disulfides from a PDB entry, and sets 
    up a Disulfide instance.
    test_name: Tests that the name attribute of the Disulfide instance is correctly set.
    test_extract: Tests that the name of the first disulfide in the loaded list matches 
    the expected value.
    test_dihedrals: Placeholder for testing the dihedral angles of the first disulfide in the loaded list.
"""

# plyint: disable=C0115
# plyint: disable=C0116
# plyint: disable=C0103


import tempfile
import unittest

from scipy.optimize import minimize

from proteusPy import Disulfide, disulfide_energy_function, load_disulfides_from_id
from proteusPy.ProteusGlobals import DATA_DIR


class TestDisulfide(unittest.TestCase):

    def setUp(self):

        entry = "5rsa"
        ok = False

        self.sslist = load_disulfides_from_id(entry, pdb_dir=DATA_DIR)
        if len(self.sslist) > 0:
            ok = True
        else:
            ok = False

        self.disulfide = Disulfide(name="tst")
        self.assertEqual(ok, True)

    def test_name(self):
        result = self.disulfide.name
        expected_result = "tst"
        self.assertEqual(result, expected_result)

    def test_extract(self):
        ss1 = self.sslist[0]
        result = ss1.name
        expected_result = "5rsa_26A_84A"
        self.assertEqual(result, expected_result)

    def test_dihedrals(self):
        from numpy.testing import assert_allclose

        ss1 = self.sslist[0]
        result = ss1.dihedrals
        expected_result = [
            -68.64177700691641,
            -87.08310517280916,
            -81.44489804423738,
            -50.83886936309293,
            -66.09666929641922,
        ]
        assert_allclose(result, expected_result, rtol=1e-05, atol=1e-08)

    def test_energy(self):

        dihedrals = [-60.0, -60.0, -90.0, -60.0, -90.0]
        result = disulfide_energy_function(dihedrals)
        expected_result = 2.5999999999999996

        self.assertEqual(result, expected_result)

    def test_minimize(self):
        # initial guess for chi1, chi2, chi3, chi4, chi5
        initial_guess = [
            -60.0,
            -60.0,
            -90.0,
            -60.0,
            -60.0,
        ]

        result = minimize(
            disulfide_energy_function, initial_guess, method="Nelder-Mead"
        )
        minimum_energy = result.fun
        expected_result = 0.4889387355489303
        self.assertEqual(minimum_energy, expected_result)

    def test_load(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            entry = "5rsa"

            sslist = load_disulfides_from_id(entry, pdb_dir=DATA_DIR)
            self.assertTrue(len(sslist) > 0)

    def test_compare(self):

        ss1 = self.sslist[0]

        self.assertTrue(ss1 == ss1)

    def test_compare2(self):

        ss1 = self.sslist[0]
        ss2 = self.sslist[1]

        self.assertFalse(ss1 == ss2)


if __name__ == "__main__":
    unittest.main()
