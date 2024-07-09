import unittest
import warnings

import numpy as np
from Bio.PDB import PDBList

from proteusPy import Disulfide, Disulfide_Energy_Function, check_header_from_file


class TestDisulfide(unittest.TestCase):

    def setUp(self):
        import tempfile

        from proteusPy.DisulfideList import load_disulfides_from_id

        entry = "5rsa"
        ok = False

        temp_dir = tempfile.TemporaryDirectory()
        pdb_home = f"{temp_dir.name}/"

        pdblist = PDBList(verbose=False)
        if not pdblist.retrieve_pdb_file(entry, file_format="pdb", pdir=pdb_home):
            ok = False
        else:
            self.sslist = load_disulfides_from_id(entry, pdb_dir=pdb_home)
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
        result = Disulfide_Energy_Function(dihedrals)
        expected_result = 2.5999999999999996

        self.assertEqual(result, expected_result)

    def test_minimize(self):
        from scipy.optimize import minimize

        from proteusPy import Disulfide_Energy_Function

        # initial guess for chi1, chi2, chi3, chi4, chi5
        initial_guess = [
            -60.0,
            -60.0,
            -90.0,
            -60.0,
            -60.0,
        ]

        result = minimize(
            Disulfide_Energy_Function, initial_guess, method="Nelder-Mead"
        )
        minimum_energy = result.fun
        expected_result = 0.4889387355489303
        self.assertEqual(minimum_energy, expected_result)

    def test_header(self):
        import tempfile

        from Bio.PDB import PDBList  # Assuming BioPython is used

        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_home = f"{temp_dir}/"
            entry = "5rsa"
            pdblist = PDBList(verbose=False)
            ok = False
            if not pdblist.retrieve_pdb_file(entry, file_format="pdb", pdir=pdb_home):
                self.fail("PDB file retrieval failed")
            else:
                filename = f"{pdb_home}pdb5rsa.ent"
                ok = check_header_from_file(filename)
        self.assertTrue(ok)

    def test_load(self):
        import tempfile

        from Bio.PDB import PDBList

        from proteusPy import DisulfideList, load_disulfides_from_id

        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_home = f"{temp_dir}/"
            entry = "5rsa"
            pdblist = PDBList(verbose=False)
            if not pdblist.retrieve_pdb_file(entry, file_format="pdb", pdir=pdb_home):
                self.fail("PDB file retrieval failed")
            else:
                sslist = load_disulfides_from_id(entry, pdb_dir=pdb_home)
                self.assertTrue(len(sslist) > 0)

    def test_compare(self):
        diff = 1.0

        ss1 = self.sslist[0]

        self.assertTrue(ss1 == ss1)

    def test_compare2(self):

        ss1 = self.sslist[0]
        ss2 = self.sslist[1]

        self.assertFalse(ss1 == ss2)


if __name__ == "__main__":
    unittest.main()
