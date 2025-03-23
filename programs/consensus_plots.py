#!/usr/bin/env python3
"""
Test script to demonstrate the box plot functionality for energy distribution by class.

This script loads the DisulfideClassGenerator, generates disulfides for a few classes,
and creates a box plot showing the energy distribution by class.

Author: Eric G. Suchanek, PhD
"""

import os
from pathlib import Path

from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator

HOME = Path.home()
PDB = Path(os.getenv("PDB", HOME / "pdb"))

MODEL_DIR = PDB / "good"

PDB_DATA_DIR = PDB / "data"


SAVE_DIR = (
    HOME / "repos" / "proteusPy_priv" / "Disulfide_Chapter/SpringerBookChapter/Figures"
)


def main():
    """Main function to demonstrate the energy box plot functionality."""
    print("Creating DisulfideClassGenerator...")
    generator = DisulfideClassGenerator(verbose=True)

    generator.generate_for_all_classes()

    generator.plot_torsion_distance_by_class(
        base=8,
        title="Torsion Distance Distribution",
        split=True,
        max_classes_per_plot=85,
        verbose=True,
        save=True,
        savedir=SAVE_DIR,
        dpi=600,
        suffix="png",
        theme="light",
    )

    # Create box plot for octant classes with automatic splitting
    print("Creating box plot for octant classes with splitting...")
    generator.plot_energy_by_class(
        base=8,
        title="Energy Distribution",
        split=False,
        max_classes_per_plot=340,
        verbose=True,
        save=True,
        savedir=SAVE_DIR,
        dpi=600,
        suffix="png",
        theme="light",
    )

    generator.plot_energy_by_class(
        base=8,
        title="Energy Distribution",
        split=True,
        max_classes_per_plot=85,  # Smaller value to demonstrate splitting
        verbose=True,
        save=True,
        savedir=SAVE_DIR,
        dpi=600,
        suffix="png",
        theme="light",
    )

    generator.plot_energy_by_class(
        base=2,
        title="Energy Distribution",
        split=False,
        verbose=True,
        save=True,
        savedir=SAVE_DIR,
        dpi=600,
        suffix="png",
        theme="light",
    )


print("Done!")


if __name__ == "__main__":
    main()
