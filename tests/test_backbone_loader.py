#!/usr/bin/env python3
# test_backbone_loader.py
# Copyright (c) 2026 Eric G. Suchanek, PhD, Flux-Frontiers
# https://github.com/Flux-Frontiers
# License: BSD
# Last revised: 2026-05-08 -egs-

"""
Unit tests for BackboneLoader and BackboneResidue.

Uses pdb5rsa.ent (RNase A, 124 residues, single chain A) from the package
data directory — the same fixture used by test_disulfide.py.
"""

# pylint: disable=C0115,C0116,C0103

import math
import unittest

from proteusPy import BackboneLoader, BackboneResidue
from proteusPy.ProteusGlobals import DATA_DIR


class TestBackboneResidue(unittest.TestCase):
    """BackboneResidue dataclass basics."""

    def test_construction(self):
        r = BackboneResidue(
            phi=-60.0,
            psi=-40.0,
            omega=180.0,
            residue_name="ALA",
            chain_id="A",
            seq_pos=5,
            pdb_id="test",
        )
        self.assertEqual(r.phi, -60.0)
        self.assertEqual(r.psi, -40.0)
        self.assertEqual(r.omega, 180.0)
        self.assertEqual(r.residue_name, "ALA")
        self.assertEqual(r.chain_id, "A")
        self.assertEqual(r.seq_pos, 5)
        self.assertEqual(r.pdb_id, "test")
        self.assertEqual(r.secondary_structure, "U")

    def test_secondary_structure_override(self):
        r = BackboneResidue(
            phi=math.nan,
            psi=math.nan,
            omega=math.nan,
            residue_name="GLY",
            chain_id="B",
            seq_pos=1,
            pdb_id="test",
            secondary_structure="H",
        )
        self.assertEqual(r.secondary_structure, "H")


class TestBackboneLoader5RSA(unittest.TestCase):
    """Integration tests against pdb5rsa.ent (RNase A)."""

    @classmethod
    def setUpClass(cls):
        loader = BackboneLoader(pdb_dir=DATA_DIR, workers=1)
        cls.residues = loader.load(pdb_ids=["5rsa"])

    # ── basic load ────────────────────────────────────────────────────────────

    def test_loads_nonzero_residues(self):
        # 5RSA has 124 residues; expect at least 100 with complete backbone
        self.assertGreater(len(self.residues), 100)

    def test_pdb_id_set_correctly(self):
        for r in self.residues:
            self.assertEqual(r.pdb_id, "5rsa")

    def test_residue_name_three_letter(self):
        for r in self.residues:
            self.assertTrue(
                1 <= len(r.residue_name) <= 3,
                f"Unexpected residue name: {r.residue_name!r}",
            )

    def test_chain_id_is_string(self):
        for r in self.residues:
            self.assertIsInstance(r.chain_id, str)

    # ── terminal angle NaN invariants ─────────────────────────────────────────

    def test_first_residue_phi_nan(self):
        """N-terminal residue of each chain must have phi=nan."""
        by_chain: dict = {}
        for r in self.residues:
            by_chain.setdefault(r.chain_id, []).append(r)
        for res_list in by_chain.values():
            first = min(res_list, key=lambda r: r.seq_pos)
            self.assertTrue(
                math.isnan(first.phi),
                f"Expected phi=nan at chain N-terminus, got {first.phi}",
            )

    def test_last_residue_psi_nan(self):
        """C-terminal residue of each chain must have psi=nan."""
        by_chain: dict = {}
        for r in self.residues:
            by_chain.setdefault(r.chain_id, []).append(r)
        for res_list in by_chain.values():
            last = max(res_list, key=lambda r: r.seq_pos)
            self.assertTrue(
                math.isnan(last.psi),
                f"Expected psi=nan at chain C-terminus, got {last.psi}",
            )

    def test_first_residue_omega_nan(self):
        """First residue of each chain must have omega=nan."""
        by_chain: dict = {}
        for r in self.residues:
            by_chain.setdefault(r.chain_id, []).append(r)
        for res_list in by_chain.values():
            first = min(res_list, key=lambda r: r.seq_pos)
            self.assertTrue(
                math.isnan(first.omega),
                f"Expected omega=nan for first residue, got {first.omega}",
            )

    # ── angle range sanity ────────────────────────────────────────────────────

    def test_interior_phi_in_range(self):
        for r in self.residues:
            if not math.isnan(r.phi):
                self.assertGreater(r.phi, -181.0, f"phi out of range for {r}")
                self.assertLessEqual(r.phi, 180.0, f"phi out of range for {r}")

    def test_interior_psi_in_range(self):
        for r in self.residues:
            if not math.isnan(r.psi):
                self.assertGreater(r.psi, -181.0, f"psi out of range for {r}")
                self.assertLessEqual(r.psi, 180.0, f"psi out of range for {r}")

    def test_interior_omega_in_range(self):
        for r in self.residues:
            if not math.isnan(r.omega):
                self.assertGreater(r.omega, -181.0, f"omega out of range for {r}")
                self.assertLessEqual(r.omega, 180.0, f"omega out of range for {r}")

    # ── secondary structure annotation ────────────────────────────────────────

    def test_secondary_structure_valid_codes(self):
        valid = {"H", "E", "P", "C", "U"}
        for r in self.residues:
            self.assertIn(
                r.secondary_structure,
                valid,
                f"Unexpected ss code {r.secondary_structure!r}",
            )

    def test_helix_residues_annotated(self):
        """5RSA has known α-helices; expect at least some 'H' annotations."""
        helix_count = sum(1 for r in self.residues if r.secondary_structure == "H")
        self.assertGreater(helix_count, 0, "No helix residues found — HELIX parsing failed")

    # ── α-helix angle statistics ──────────────────────────────────────────────

    def test_helix_phi_psi_near_ideal(self):
        """Mean φ/ψ of helical interior residues should cluster near ideal α-helix.

        Ideal: φ ≈ −60°, ψ ≈ −40°.  Allow ±20° tolerance for real structures.
        """
        helix_interior = [
            r
            for r in self.residues
            if r.secondary_structure == "H" and not math.isnan(r.phi) and not math.isnan(r.psi)
        ]
        if not helix_interior:
            self.skipTest("No annotated helical interior residues to check")

        avg_phi = sum(r.phi for r in helix_interior) / len(helix_interior)
        avg_psi = sum(r.psi for r in helix_interior) / len(helix_interior)

        self.assertAlmostEqual(
            avg_phi, -60.0, delta=25.0, msg=f"avg φ={avg_phi:.1f}° far from ideal"
        )
        self.assertAlmostEqual(
            avg_psi, -40.0, delta=30.0, msg=f"avg ψ={avg_psi:.1f}° far from ideal"
        )

    # ── BackboneResidue attribute interface (WaveRider duck-type) ─────────────

    def test_waverider_interface(self):
        """Every residue must expose the attributes BackboneAngleList.from_proteuspy() expects."""
        required = (
            "phi",
            "psi",
            "omega",
            "residue_name",
            "chain_id",
            "seq_pos",
            "pdb_id",
            "secondary_structure",
        )
        for r in self.residues[:10]:
            for attr in required:
                self.assertTrue(hasattr(r, attr), f"Missing attribute: {attr}")


class TestBackboneLoaderEmptyResult(unittest.TestCase):
    """Edge-case: nonexistent pdb_id returns empty list without crashing."""

    def test_missing_file_returns_empty(self):
        loader = BackboneLoader(pdb_dir=DATA_DIR, workers=1)
        result = loader.load(pdb_ids=["0000"])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
