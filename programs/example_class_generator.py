#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the DisulfideClassGenerator class.

This script shows how to:
1. Create a DisulfideClassGenerator instance
2. Load a CSV file with class metrics
3. Generate disulfides for specific classes
4. Generate disulfides for multiple classes
5. Save the generated disulfides to files

Author: Eric G. Suchanek, PhD
Last Modification: 2025-03-15
"""

import os
import pickle
import sys
from pathlib import Path

# Add the parent directory to the Python path to import proteusPy modules
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteusPy.generate_class_disulfides import DisulfideClassGenerator


def main():
    """
    Main function demonstrating how to use the DisulfideClassGenerator class.
    """
    # Path to the CSV file
    csv_file = "~/repos/proteusPy/binary_class_metrics_0.00.csv"

    print("Creating DisulfideClassGenerator instance...")

    # Create a generator instance and load the CSV file
    generator = DisulfideClassGenerator(csv_file)

    # Example 1: Generate disulfides for a specific class using class_str
    class_str = "++++++"  # The RH Spiral class - this should be "+++++", not "+++++"
    print(f"\nGenerating disulfides for class {class_str}...")

    disulfide_list = generator.generate_for_class(class_str, use_class_str=True)

    if disulfide_list:
        print(f"Generated {len(disulfide_list)} disulfides for class {class_str}.")
        print(f"List name: {disulfide_list.pdb_id}")

        # Display information about the first few disulfides
        print("\nFirst 3 disulfides:")
        for i, disulfide in enumerate(disulfide_list[:3]):
            print(f"Disulfide {i+1}:")
            print(f"  Name: {disulfide.name}")
            print(f"  Dihedrals: {disulfide.dihedrals}")
            print(f"  Energy: {disulfide.energy:.2f} kcal/mol")
            print(f"  Torsion length: {disulfide.torsion_length:.2f}")
            print(f"  Cα distance: {disulfide.ca_distance:.2f} Å")
            print()

        # Save the disulfide list to a file
        output_file = f"class_{class_str}_disulfides.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(disulfide_list, f)
        print(f"Saved disulfides to {output_file}.")
    else:
        print(f"Class {class_str} not found. Let's try with '+++++' instead.")
        class_str = "++++++"  # Corrected class string
        disulfide_list = generator.generate_for_class("+++++", use_class_str=True)
        if disulfide_list:
            print(
                f"Generated {len(disulfide_list)} disulfides for class '+++++' (RH Spiral)."
            )

    # Example 2: Generate disulfides for a specific class using class ID
    class_id = "-----"  # The LH Spiral class
    print(f"\nGenerating disulfides for class ID {class_id}...")

    disulfide_list = generator.generate_for_class(class_id, use_class_str=True)

    if disulfide_list:
        print(f"Generated {len(disulfide_list)} disulfides for class ID {class_id}.")
        print(f"List name: {disulfide_list.pdb_id}")

        # Display information about the first disulfide
        first_disulfide = disulfide_list[0]
        print(f"First disulfide: {first_disulfide.name}")
        print(f"Dihedrals: {first_disulfide.dihedrals}")
        print(f"Energy: {first_disulfide.energy:.2f} kcal/mol")

    # Example 3: Generate disulfides for multiple classes
    selected_classes = [
        "+++++",
        "-----",
        "-+---",
    ]  # RH Spiral, LH Spiral, and RH Staple
    print(f"\nGenerating disulfides for selected classes: {selected_classes}...")

    class_disulfides = generator.generate_for_selected_classes(
        selected_classes, use_class_str=True
    )

    print("\nGenerated disulfides for the following classes:")
    for class_id, disulfide_list in class_disulfides.items():
        print(
            f"Class {class_id}: {len(disulfide_list)} disulfides, Avg Energy: {disulfide_list.average_energy:.2f} kcal/mol"
        )

    # Example 4: Find the minimum energy disulfide for each class
    print("\nMinimum energy disulfides for each class:")
    for class_id, disulfide_list in class_disulfides.items():
        min_energy_disulfide = min(disulfide_list, key=lambda ss: ss.energy)
        print(
            f"Class {class_id}: {min_energy_disulfide.energy:.2f} kcal/mol, Dihedrals: {min_energy_disulfide.dihedrals}"
        )

    print("\nExample complete.")


if __name__ == "__main__":
    main()
