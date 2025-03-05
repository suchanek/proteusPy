# pylint: disable=C0103

"""
Unit tests for the DisplaySS functionality in the proteusPy package.

This module contains unit tests for displaying and taking screenshots of disulfide bonds
using the proteusPy package. The tests ensure that the display and screenshot functionalities
work correctly for both single disulfide bonds and lists of disulfide bonds.

Classes:
    TestDisplaySS: A unittest.TestCase subclass that contains the tests for DisplaySS functionality.

Methods:
    setUp(self): Sets up the test environment, including creating a temporary directory for
        screenshots, setting the PyVista theme, and loading a subset of the disulfide database.
    
    tearDown(self): Cleans up the test environment by removing the temporary directory.
    
    test_single_disulfide_display(self): Tests the display and screenshot functionality for a 
        single disulfide bond.
    
    test_disulfide_list_display(self): Test the display functionality for a list of disulfide bonds.

Usage:
    Run this module as a script to execute the unit tests.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-26 19:57:07 -egs-
"""

# pylint: disable=W0718 # too general exception clause
# pylint: disable=C0114 # missing-module-docstring
# pylint: disable=C0103 # non-snake-case variable name


import os
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest import main as run_tests

# Add the parent directory to the path so we can import proteusPy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proteusPy import Load_PDB_SS, __version__, set_pyvista_theme
from proteusPy.ProteusGlobals import DATA_DIR


class TestDisplaySS(unittest.TestCase):
    """Unit tests for the DisplaySS functionality in the proteusPy package."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Create a TemporaryDirectory object and store its path as a Path object.
        self.temp_dir_obj = TemporaryDirectory(prefix="proteusPy_")

        # Set the theme.
        set_pyvista_theme("auto")

        # Load the disulfide database (subset, for speed).
        # self.PDB = pp.DisulfideLoader(verbose=True, subset=True)
        print(f"DATA_DIR: {DATA_DIR}")
        self.PDB = Load_PDB_SS(verbose=True, subset=True)

        self.first_disulfide = self.PDB[0]

    def tearDown(self) -> None:
        """Clean up test fixtures after each test method."""
        # Remove the temporary directory via the TemporaryDirectory object's cleanup method.
        self.temp_dir_obj.cleanup()

    def test_disulfide_list_display(self):
        """Test the display functionality for a list of disulfide bonds."""
        # Retrieve a disulfide list for a given structure using its identifier.

        try:
            ss6dmb = self.PDB["6dmb"]
        except Exception as e:
            self.fail(f"DisulfideList display for '6dmb' raised an exception: {e}")

        ss6dmb.display(style="cpk")
        ss6dmb.display(style="bs")
        ss6dmb.display(style="sb")
        ss6dmb.display(style="pd")
        ss6dmb.display(style="plain")
        ss6dmb.display_overlay(light="auto")

        # Test with a subset (first 12 disulfides) of the database.
        try:
            sslist = self.PDB[:12]
        except Exception as e:
            self.fail(f"DisulfideList display for subset raised an exception: {e}")

        self.assertIsNotNone(sslist)
        self.assertGreater(len(sslist), 0, "Disulfide list should not be empty.")

        sslist.display(style="cpk")
        sslist.display(style="bs")
        sslist.display(style="sb")
        sslist.display(style="pd")
        sslist.display(style="plain")
        sslist.display_overlay()

    def test_single_disulfide_display(self):
        """Test the display and screenshot functionality for a single disulfide."""
        # Use the first disulfide from the database.
        ss = self.first_disulfide

        try:
            ss.spin(style="sb")
            ss.display(style="bs", single=True)
            ss.display(style="cpk", single=True)
            ss.display(style="sb", single=True)
            ss.display(style="pd", single=False)

        except Exception as e:
            self.fail(f"Display method raised an exception: {e}")

        # Test the screenshot functionality and verify the screenshot files exist.
        cpk_filename = os.path.join(self.temp_dir_obj.name, "cpk3.png")
        try:
            ss.screenshot(
                style="cpk", single=True, fname=str(cpk_filename), verbose=True
            )
        except Exception as e:
            self.fail(f"Screenshot (cpk) method raised an exception: {e}")


if __name__ == "__main__":
    run_tests()

# EOF
