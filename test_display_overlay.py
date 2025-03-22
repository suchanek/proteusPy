#!/usr/bin/env python
"""
Test script to evaluate the display_overlay function with different parameters.
"""

from proteusPy import Load_PDB_SS

# Load a subset of PDB disulfides
print("Loading PDB disulfides...")
PDB_SS = Load_PDB_SS(verbose=True, subset=True)

ss = PDB_SS[0]
ss.spin(style="sb")
# Get a small subset of disulfides for testing
sslist = PDB_SS[:5]

# Test 1: Basic display without spin
print("\nTest 1: Basic display without spin")
# sslist.display_overlay(verbose=True)

# Test 2: Display with spin=True
print("\nTest 2: Display with spin=True")
sslist.display_overlay(spin=True, verbose=True)

# Test 3: Display with screenshot
print("\nTest 3: Display with screenshot")
# sslist.display_overlay(
#    screenshot=True, fname="test_overlay.png", verbose=True
# )

print("All tests completed.")
