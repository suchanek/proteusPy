"""
Example script demonstrating how to use the disulfide_schematic module
to create publication-ready 2D diagrams of disulfide bonds.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-03
"""

import os
import sys

import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import proteusPy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import proteusPy as pp
from proteusPy.disulfide_schematic import (
    create_disulfide_schematic,
    create_disulfide_schematic_from_model,
)


def main():
    """Main function demonstrating disulfide schematic creation."""
    # Create output directory if it doesn't exist
    output_dir = "schematic_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading disulfide database (subset)...")
    # Load a disulfide from the database
    PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)

    # Example 1: Create a schematic from a real disulfide
    print("\nExample 1: Creating schematic from a real disulfide...")
    ss = PDB_SS[0]  # Get the first disulfide
    print(
        f"Selected disulfide: {ss.pdb_id} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}"
    )

    output_file = os.path.join(output_dir, "real_disulfide_schematic.png")
    fig1, ax1 = create_disulfide_schematic(
        disulfide=ss, output_file=output_file, show_angles=True, style="publication"
    )
    print(f"Saved schematic to: {output_file}")

    # Example 2: Create a schematic from a model disulfide
    print("\nExample 2: Creating schematic from a model disulfide...")
    output_file = os.path.join(output_dir, "model_disulfide_schematic.png")
    fig2, ax2 = create_disulfide_schematic_from_model(
        chi1=-60,
        chi2=-60,
        chi3=-90,
        chi4=-60,
        chi5=-60,
        output_file=output_file,
        show_angles=True,
    )
    print(f"Saved schematic to: {output_file}")

    # Example 3: Different styles
    print("\nExample 3: Creating schematics with different styles...")

    # Simple style
    output_file = os.path.join(output_dir, "simple_style_schematic.png")
    create_disulfide_schematic(disulfide=ss, output_file=output_file, style="simple")
    print(f"Saved simple style schematic to: {output_file}")

    # Detailed style
    output_file = os.path.join(output_dir, "detailed_style_schematic.png")
    create_disulfide_schematic(
        disulfide=ss, output_file=output_file, style="detailed", show_angles=True
    )
    print(f"Saved detailed style schematic to: {output_file}")

    # Example 4: Different output formats
    print("\nExample 4: Creating schematics in different formats...")

    # SVG format
    output_file = os.path.join(output_dir, "disulfide_schematic.svg")
    create_disulfide_schematic(disulfide=ss, output_file=output_file)
    print(f"Saved SVG schematic to: {output_file}")

    # PDF format
    output_file = os.path.join(output_dir, "disulfide_schematic.pdf")
    create_disulfide_schematic(disulfide=ss, output_file=output_file)
    print(f"Saved PDF schematic to: {output_file}")

    print("\nAll examples completed successfully!")
    print(f"Output files can be found in the '{output_dir}' directory.")


if __name__ == "__main__":
    main()
