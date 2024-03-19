import unittest

import numpy as np
from Bio.PDB.vectors import Vector
from numpy.testing import assert_array_equal

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS

# Define a _tolerance threshold
_tolerance = 1e-8


def cmp_vec(v1: Vector, v2: Vector, tol: float) -> bool:
    "Return true if the length of the difference between the two vectors is less than a tolerance."
    _diff = v2 - v1
    _len = _diff.norm()
    return _len < tol


class TestDisulfide(unittest.TestCase):

    def setUp(self):
        self.PDB_SS = Load_PDB_SS(verbose=False, subset=True)
        self.disulfide = Disulfide(name="tst")

    def test_name(self):
        result = self.disulfide.name
        expected_result = "tst"
        self.assertEqual(result, expected_result)

    def test_extract(self):
        ss1 = self.PDB_SS[0]
        result = ss1.name
        expected_result = "4yys_22A_65A"
        self.assertEqual(result, expected_result)

    def test_dihedrals(self):
        ss1 = self.PDB_SS[0]
        result = ss1.dihedrals
        expected_result = [
            174.62923341948851,
            82.51771039903726,
            -83.32224872066772,
            -62.52364351964355,
            -73.82728569383424,
        ]
        self.assertEqual(result, expected_result)

    def test_energy(self):
        from proteusPy.Disulfide import Disulfide_Energy_Function

        dihedrals = [-60.0, -60.0, -90.0, -60.0, -90.0]
        result = Disulfide_Energy_Function(dihedrals)
        expected_result = 2.5999999999999996

        self.assertEqual(result, expected_result)

    def test_minimize(self):
        from scipy.optimize import minimize

        from proteusPy.Disulfide import Disulfide_Energy_Function

        # initial guess for chi1, chi2, chi3, chi4, chi5
        initial_guess = [
            -60.0,
            -60.0,
            -90.0,
            -60.0,
            -60.0,
        ]

        dihedrals = [-60.0, -60.0, -90.0, -60.0, -90.0]
        result = minimize(
            Disulfide_Energy_Function, initial_guess, method="Nelder-Mead"
        )
        minimum_energy = result.fun
        expected_result = 0.4889387355489303
        self.assertEqual(minimum_energy, expected_result)

    def test_header(self):
        from proteusPy.Disulfide import check_header_from_file

        filename = "../data/pdb5rsa.ent"
        ok = False
        ok = check_header_from_file(filename)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
