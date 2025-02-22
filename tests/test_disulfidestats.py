"""
This module contains unit tests for the DisulfideStats class in proteusPy.
It includes tests for methods that build dataframes and calculate statistics
related to disulfide bonds.
Last Revision: 2025-02-18 23:16:37 -egs-
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from proteusPy import DisulfideStats, load_disulfides_from_id
from proteusPy.ProteusGlobals import DATA_DIR


class TestDisulfideStats(unittest.TestCase):
    """Test the DisulfideStats class."""

    def setUp(self):
        """Set up mock Disulfide objects for testing."""
        self.mock_sslist = load_disulfides_from_id("5rsa", pdb_dir=DATA_DIR)

    def test_build_distance_df(self):
        """Test the build_distance_df method."""
        df = DisulfideStats.build_distance_df(self.mock_sslist)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.mock_sslist))
        self.assertListEqual(
            list(df.columns),
            [
                "source",
                "ss_id",
                "proximal",
                "distal",
                "energy",
                "ca_distance",
                "cb_distance",
                "sg_distance",
            ],
        )

    def test_build_torsion_df(self):
        """Test the build_torsion_df method."""
        df = DisulfideStats.build_torsion_df(self.mock_sslist)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.mock_sslist))
        self.assertIn("chi1", df.columns)
        self.assertIn("torsion_length", df.columns)

    def test_create_deviation_dataframe(self):
        """Test the create_deviation_dataframe method."""
        df = DisulfideStats.create_deviation_dataframe(self.mock_sslist)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("PDB_ID", df.columns)
        self.assertEqual(len(df), len(self.mock_sslist))

    def test_calculate_torsion_statistics(self):
        """Test the calculate_torsion_statistics method."""
        tor_stats, dist_stats = DisulfideStats.calculate_torsion_statistics(
            self.mock_sslist
        )
        self.assertIsInstance(tor_stats, pd.DataFrame)
        self.assertIsInstance(dist_stats, pd.DataFrame)
        self.assertIn("chi1", tor_stats.columns)
        self.assertIn("ca_distance", dist_stats.columns)

    def test_extract_distances(self):
        """Test the extract_distances method."""
        sg_distances = DisulfideStats.extract_distances(
            self.mock_sslist, distance_type="sg"
        )
        ca_distances = DisulfideStats.extract_distances(
            self.mock_sslist, distance_type="ca"
        )

        self.assertEqual(len(sg_distances), len(self.mock_sslist))
        self.assertEqual(len(ca_distances), len(self.mock_sslist))

        filtered_sg_distances = DisulfideStats.extract_distances(
            self.mock_sslist, distance_type="sg", comparison="less", cutoff=4.0
        )

        self.assertTrue(all(d <= 4.0 for d in filtered_sg_distances))

    def test_bond_angle_ideality(self):
        """Test the bond_angle_ideality method."""
        ss_mock = MagicMock()
        ss_mock.coords_array = np.random.rand(12, 3)  # Mock coordinates for atoms
        ss_mock.quiet = True

        rms_diff = DisulfideStats.bond_angle_ideality(ss_mock)

        # Assert that RMS difference is a float and non-negative
        self.assertIsInstance(rms_diff, float)
        self.assertGreaterEqual(rms_diff, 0)

    def test_bond_length_ideality(self):
        """Test the bond_length_ideality method."""
        ss_mock = MagicMock()
        ss_mock.coords_array = np.random.rand(12, 3)  # Mock coordinates for atoms
        ss_mock.quiet = True

        rms_diff = DisulfideStats.bond_length_ideality(ss_mock)

        # Assert that RMS difference is a float and non-negative
        self.assertIsInstance(rms_diff, float)
        self.assertGreaterEqual(rms_diff, 0)

    def test_calculate_std_cutoff(self):
        """Test the calculate_std_cutoff method."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})

        cutoff = DisulfideStats.calculate_std_cutoff(df, column="values", num_std=2)

        mean_val, std_val = df["values"].mean(), df["values"].std()

        expected_cutoff = mean_val + (2 * std_val)

        self.assertAlmostEqual(cutoff, expected_cutoff)

    def test_calculate_percentile_cutoff(self):
        """Test the calculate_percentile_cutoff method."""
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})

        cutoff_95th_percentile = DisulfideStats.calculate_percentile_cutoff(
            df, column="values", percentile=95
        )

        expected_cutoff_95th_percentile = np.percentile(df["values"], 95)

        self.assertAlmostEqual(cutoff_95th_percentile, expected_cutoff_95th_percentile)


if __name__ == "__main__":
    unittest.main()
