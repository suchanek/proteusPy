#!/usr/bin/env python3
"""
Test script to verify font fallback functionality in DisulfideVisualization.

This script tests the display of both single disulfides and disulfide lists
to verify that text rendering works correctly on Linux systems.

Author: Eric G. Suchanek, PhD
"""

import os
import platform
import sys

from proteusPy import Load_PDB_SS, __version__
from proteusPy.utility import find_arial_font


def main():
    """Test font fallback functionality."""
    print(f"proteusPy version: {__version__}")
    print(f"Platform: {platform.system()}")

    # Test font finding
    font_path = find_arial_font()
    if font_path:
        print(f"Found font: {font_path}")
    else:
        print("No suitable font found. Will use fallback estimation.")

    # Load a subset of the disulfide database
    print("Loading PDB_SS subset...")
    PDB = Load_PDB_SS(verbose=True, subset=True)

    # Test single disulfide display
    print("\nTesting single disulfide display...")
    ss = PDB[0]
    print(
        f"Displaying single disulfide: {ss.pdb_id} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}"
    )
    ss.display(style="sb", single=True)

    # Test disulfide list display
    print("\nTesting disulfide list display...")
    sslist = PDB[:4]  # Get first 4 disulfides
    print(f"Displaying disulfide list with {len(sslist)} members")
    sslist.display(style="sb")

    # Test overlay display
    print("\nTesting overlay display...")
    sslist.display_overlay()

    print(
        "\nTests completed. If you can see text titles in all visualizations, the fix was successful."
    )


if __name__ == "__main__":
    main()
