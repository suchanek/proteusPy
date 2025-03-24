#!/usr/bin/env python
"""
A short program to test the proteusPy.DisulfideLoader.display_overlay() function.
This script loads the PDB SS database and displays an overlay of disulfides
for a specified PDB ID.
"""

import logging
import os
import sys
from pathlib import Path

# Import proteusPy modules
import proteusPy as pp
from proteusPy.logger_config import create_logger


def main():
    # Set up logging
    logger = create_logger("test_display_overlay")
    logger.setLevel(logging.INFO)

    # Load the PDB SS database (using subset for faster loading)
    print("Loading PDB SS database (subset)...")
    PDB_SS = pp.Load_PDB_SS(verbose=True, subset=True)

    # Display information about the loaded database
    print("\nDatabase information:")
    PDB_SS.describe()

    # Get a list of available PDB IDs (first 5)
    pdb_ids = PDB_SS.IDList[:5]
    print(f"\nFirst 5 available PDB IDs: {pdb_ids}")

    # Choose a PDB ID to display
    pdb_id = pdb_ids[0]  # Use the first available PDB ID

    # Display the overlay for the selected PDB ID
    print(f"\nDisplaying overlay for PDB ID: {pdb_id}")
    PDB_SS.display_overlay(pdb_id, verbose=True)

    # Also demonstrate using a slice of the loader
    print("\nDisplaying overlay for first 5 disulfides in the database")
    PDB_SS[:5].display_overlay(verbose=True, spin=True)


if __name__ == "__main__":
    main()
