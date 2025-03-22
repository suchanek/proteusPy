#!/usr/bin/env python
"""
Test script to verify the spin=True functionality in display_overlay.
"""

from proteusPy import Load_PDB_SS

# Load a subset of PDB disulfides
PDB_SS = Load_PDB_SS(verbose=True, subset=True)

# Get a small subset of disulfides for testing
sslist = PDB_SS[:10]

# Try to display the overlay with spin=True
print("Displaying overlay with spin=True...")
sslist.display_overlay(spin=True)
print("Display completed.")
