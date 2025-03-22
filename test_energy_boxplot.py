#!/usr/bin/env python3
"""
Test script to demonstrate the box plot functionality for energy distribution by class.

This script loads the DisulfideClassGenerator, generates disulfides for a few classes,
and creates a box plot showing the energy distribution by class.

Author: Eric G. Suchanek, PhD
"""

import sys
from pathlib import Path

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator


def main():
    """Main function to demonstrate the energy box plot functionality."""
    print("Creating DisulfideClassGenerator...")
    generator = DisulfideClassGenerator(verbose=True)

    # Generate disulfides for specific octant classes that we know exist
    print("Generating disulfides for specific classes...")

    # These are octant classes that should exist in the dataset
    octant_classes = ["12222", "12232", "12322"]
    print(f"Generating disulfides for octant classes: {octant_classes}")

    generator.generate_for_selected_classes(octant_classes)

    # Try to generate a few binary classes with specific patterns
    binary_patterns = ["+-+-+", "+--++", "++---"]
    print(f"Generating disulfides for binary patterns: {binary_patterns}")
    generator.generate_for_selected_classes(binary_patterns)

    # Create box plot for all classes
    print("Creating box plot for all classes...")
    generator.plot_energy_by_class(title="Energy Distribution by Class")

    # Create box plot for binary classes only
    print("Creating box plot for binary classes only...")
    generator.plot_energy_by_class(base=2, title="Energy Distribution by Binary Class")

    # Create box plot for octant classes only
    print("Creating box plot for octant classes only...")
    generator.plot_energy_by_class(base=8, title="Energy Distribution by Octant Class")

    print("Done!")


if __name__ == "__main__":
    main()
