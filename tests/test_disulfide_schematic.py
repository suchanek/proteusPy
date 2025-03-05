"""
Unit tests for the disulfide_schematic module in the proteusPy package.

This module contains unit tests for creating 2D schematic diagrams of disulfide bonds
using the proteusPy package.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-03
"""

import os
import sys
import unittest
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

from proteusPy import Load_PDB_SS
from proteusPy.disulfide_schematic import (
    create_disulfide_schematic,
    create_disulfide_schematic_from_model,
)


class TestDisulfideSchematic(unittest.TestCase):
    """Unit tests for the disulfide_schematic module in the proteusPy package."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a TemporaryDirectory object and store its path
        self.temp_dir_obj = TemporaryDirectory(prefix="proteusPy_")

        # Load the disulfide database (subset, for speed)
        self.PDB = Load_PDB_SS(verbose=False, subset=True)

        # Get the first disulfide for testing
        self.first_disulfide = self.PDB[0]

    def tearDown(self) -> None:
        """Clean up test fixtures after each test method."""
        # Remove the temporary directory
        self.temp_dir_obj.cleanup()

        # Close any open matplotlib figures
        plt.close("all")

    def test_create_disulfide_schematic(self):
        """Test creating a schematic diagram from a disulfide object."""
        # Test with a real disulfide
        output_file = os.path.join(self.temp_dir_obj.name, "test_schematic.png")

        fig, ax = create_disulfide_schematic(
            disulfide=self.first_disulfide,
            output_file=output_file,
            show_angles=True,
            style="publication",
        )

        # Check that the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Check that the figure and axis were returned
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_create_disulfide_schematic_with_ca_ca_distance(self):
        """Test creating a schematic diagram with Ca-Ca distance line."""
        # Test with a real disulfide and Ca-Ca distance line
        output_file = os.path.join(self.temp_dir_obj.name, "test_schematic_ca_ca.png")

        fig, ax = create_disulfide_schematic(
            disulfide=self.first_disulfide,
            output_file=output_file,
            show_angles=True,
            show_ca_ca_distance=True,
            style="publication",
        )

        # Check that the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Check that the figure and axis were returned
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_create_disulfide_schematic_from_model(self):
        """Test creating a schematic diagram from model parameters."""
        # Test with model parameters
        output_file = os.path.join(self.temp_dir_obj.name, "test_model_schematic.png")

        fig, ax = create_disulfide_schematic_from_model(
            chi1=-60,
            chi2=-60,
            chi3=-90,
            chi4=-60,
            chi5=-60,
            output_file=output_file,
            show_angles=True,
        )

        # Check that the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Check that the figure and axis were returned
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_create_model_schematic_with_ca_ca_distance(self):
        """Test creating a model schematic diagram with Ca-Ca distance line."""
        # Test with model parameters and Ca-Ca distance line
        output_file = os.path.join(
            self.temp_dir_obj.name, "test_model_schematic_ca_ca.png"
        )

        fig, ax = create_disulfide_schematic_from_model(
            chi1=-60,
            chi2=-60,
            chi3=-90,
            chi4=-60,
            chi5=-60,
            output_file=output_file,
            show_angles=True,
            show_ca_ca_distance=True,
        )

        # Check that the output file was created
        self.assertTrue(os.path.exists(output_file))

        # Check that the figure and axis were returned
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_different_styles(self):
        """Test creating schematics with different styles."""
        # Test simple style
        simple_file = os.path.join(self.temp_dir_obj.name, "simple_style.png")
        create_disulfide_schematic(
            disulfide=self.first_disulfide, output_file=simple_file, style="simple"
        )
        self.assertTrue(os.path.exists(simple_file))

        # Test detailed style
        detailed_file = os.path.join(self.temp_dir_obj.name, "detailed_style.png")
        create_disulfide_schematic(
            disulfide=self.first_disulfide,
            output_file=detailed_file,
            style="detailed",
            show_angles=True,
        )
        self.assertTrue(os.path.exists(detailed_file))

    def test_output_formats(self):
        """Test creating schematics in different output formats."""
        # Test SVG format
        svg_file = os.path.join(self.temp_dir_obj.name, "test_schematic.svg")
        create_disulfide_schematic(disulfide=self.first_disulfide, output_file=svg_file)
        self.assertTrue(os.path.exists(svg_file))

        # Test PDF format
        pdf_file = os.path.join(self.temp_dir_obj.name, "test_schematic.pdf")
        create_disulfide_schematic(disulfide=self.first_disulfide, output_file=pdf_file)
        self.assertTrue(os.path.exists(pdf_file))


if __name__ == "__main__":
    unittest.main()
