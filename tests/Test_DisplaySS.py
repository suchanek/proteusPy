"""
Unit tests for the DisplaySS functionality in the proteusPy package.

This module contains unit tests for displaying and taking screenshots of disulfide bonds
using the proteusPy package. The tests ensure that the display and screenshot functionalities
work correctly for both single disulfide bonds and lists of disulfide bonds.
Last revision: 2025-02-08 17:08:52 -egs-
"""

import os
import shutil
import tempfile
import unittest

from proteusPy import set_pyvista_theme
from proteusPy.DisulfideLoader import Load_PDB_SS

# pylint: disable=W0718 # too general exception clause
# pylint: disable=C0114 # missing-module-docstring
# pylint: disable=C0103 # non-snake-case variable name


class TestDisplaySS(unittest.TestCase):
    """Unit tests for the DisplaySS functionality in the proteusPy package."""

    def setUp(self):
        # Create a temporary directory for storing screenshots.
        self.tmp_dir = tempfile.mkdtemp()
        # Set the theme.
        set_pyvista_theme("auto")
        # Load the disulfide database (subset, for speed).
        self.PDB = Load_PDB_SS(verbose=True, subset=True)
        self.PDB.describe()

    def tearDown(self):
        # Clean up the temporary directory.
        shutil.rmtree(self.tmp_dir)

    def test_single_disulfide_display(self):
        """Test the display and screenshot functionality for a single disulfide."""
        # Get the first disulfide from the database.
        ss = self.PDB[0]

        # Call display methods with various styles.
        try:
            ss.display(style="bs", single=True)
            ss.display(style="cpk", single=True)
            ss.display(style="sb", single=True)
            ss.display(style="pd", single=False)
        except Exception as e:
            self.fail(f"Display method raised an exception: {e}")

        # Test the screenshot functionality and verify the screenshot files exist.
        cpk_filename = os.path.join(self.tmp_dir, "cpk3.png")
        try:
            ss.screenshot(style="cpk", single=True, fname=cpk_filename, verbose=True)
        except Exception as e:
            self.fail(f"Screenshot (cpk) method raised an exception: {e}")
        self.assertTrue(
            os.path.exists(cpk_filename), "CPK screenshot file does not exist."
        )

        sb_filename = os.path.join(self.tmp_dir, "sb3.png")
        try:
            ss.screenshot(style="sb", single=False, fname=sb_filename, verbose=True)
        except Exception as e:
            self.fail(f"Screenshot (sb) method raised an exception: {e}")
        self.assertTrue(
            os.path.exists(sb_filename), "SB screenshot file does not exist."
        )

    def test_disulfide_list_display(self):
        """Test the display functionality for a list of disulfide bonds."""
        # Retrieve a disulfide list for a given structure using its identifier.
        try:
            ss4yss = self.PDB["6dmb"]
            ss4yss.display(style="cpk")
            ss4yss.display(style="bs")
            ss4yss.display(style="sb")
            ss4yss.display(style="pd")
            ss4yss.display(style="plain")
        except Exception as e:
            self.fail(f"DisulfideList display for '4yys' raised an exception: {e}")

        # Test with a subset (first 12 disulfides) of the database.
        try:
            sslist = self.PDB[:12]
            sslist.display(style="cpk")
            sslist.display(style="bs")
            sslist.display(style="sb")
            sslist.display(style="pd")
            sslist.display(style="plain")
        except Exception as e:
            self.fail(f"DisulfideList display for subset raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
