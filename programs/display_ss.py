"""
Display and screenshot functionality for disulfide bonds using the proteusPy package.

This module demonstrates the display and screenshot functionalities for disulfide bonds
using the proteusPy package. It shows various visualization methods for both single 
disulfide bonds and lists of disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-13
"""

# pylint: disable=W1203

import os
from tempfile import TemporaryDirectory

from proteusPy import Load_PDB_SS, set_pyvista_theme
from proteusPy.ProteusGlobals import DATA_DIR


def main():
    """Run the DisplaySS demonstration."""
    # Create a temporary directory for screenshots
    temp_dir_obj = TemporaryDirectory(prefix="proteusPy_")

    # Set the theme
    set_pyvista_theme("auto")

    # Load the disulfide database (subset, for speed)
    print(f"DATA_DIR: {DATA_DIR}")
    PDB = Load_PDB_SS(verbose=True, subset=True)

    first_disulfide = PDB[0]

    # Test the display functionality for a list of disulfide bonds
    try:
        ss6dmb = PDB["6dmb"]
    except Exception as e:
        print(f"DisulfideList display for '6dmb' raised an exception: {e}")
        return

    ss6dmb.display(style="cpk")
    ss6dmb.display(style="bs")
    ss6dmb.display(style="sb")
    ss6dmb.display(style="pd")
    ss6dmb.display(style="plain")
    ss6dmb.display_overlay()

    # Test with a subset (first 12 disulfides) of the database
    try:
        sslist = PDB[:12]
    except Exception as e:
        print(f"DisulfideList display for subset raised an exception: {e}")
        return

    if len(sslist) > 0:
        sslist.display(style="cpk")
        sslist.display(style="bs")
        sslist.display(style="sb")
        sslist.display(style="pd")
        sslist.display(style="plain")
        sslist.display_overlay()

    # Test single disulfide display
    ss = first_disulfide

    try:
        ss.spin(style="sb")
        ss.display(style="bs", single=True)
        ss.display(style="cpk", single=True)
        ss.display(style="sb", single=True)
        ss.display(style="pd", single=False)

    except Exception as e:
        print(f"Display method raised an exception: {e}")
        return

    # Test the screenshot functionality
    cpk_filename = os.path.join(temp_dir_obj.name, "cpk3.png")
    try:
        ss.screenshot(style="cpk", single=True, fname=str(cpk_filename), verbose=True)
    except Exception as e:
        print(f"Screenshot (cpk) method raised an exception: {e}")

    # Cleanup
    temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
