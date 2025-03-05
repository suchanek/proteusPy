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
    output_dir = (
        "/Users/egs/repos/proteusPy_priv/Disulfide_Chapter/SpringerBookChapter/Figures"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Loading disulfide database (full)...")
    # Load a disulfide from the database
    pdb = pp.Load_PDB_SS(verbose=True, subset=False)
    best_id = "2q7q_75D_140D"
    worst_id = "6vxk_801B_806B"
    worst_ss = pdb[worst_id]

    ss = pdb[best_id]


    pdbid = ss.pdb_id
    # Example 1: Create a schematic from a real disulfide
    print("\nExample 1: Creating schematic from a real disulfide...")

    print(
        f"Selected disulfide: {ss.pdb_id} {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}"
    )

    output_file = os.path.join(output_dir, "real_disulfide_schematic.png")
    create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        show_angles=True,
        show_ca_ca_distance=True,  # Show the Cα-Cα distance
        style="publication",
    )
    print(f"Saved schematic to: {output_file}")

    # Example 3: Different styles
    print("\nExample 3: Creating schematics with different styles...")

    # Simple style
    output_file = os.path.join(output_dir, f"{pdbid}_simple_style_schematic.png")
    create_disulfide_schematic(
        disulfide=ss, output_file=output_file, style="simple", show_ca_ca_distance=True
    )
    print(f"Saved simple style schematic to: {output_file}")

    # Detailed style
    output_file = os.path.join(output_dir, f"{pdbid}_detailed_style_schematic.png")
    create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        style="detailed",
        show_angles=True,
        show_ca_ca_distance=True,
    )
    print(f"Saved detailed style schematic to: {output_file}")

    # Example 4: Different output formats
    print("\nExample 4: Creating schematics in different formats...")

    # SVG format
    output_file = os.path.join(output_dir, f"{pdbid}_disulfide_schematic.svg")
    create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        style="detailed",
        show_angles=True,
        show_ca_ca_distance=True,
    )
    print(f"Saved SVG schematic to: {output_file}")

    output_file = os.path.join(output_dir, f"{pdbid}_disulfide_schematic.png")
    create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        show_angles=True,
        show_ca_ca_distance=True,
        style="detailed",
    )
    print(f"Saved PNG schematic to: {output_file}")

    # PDF format
    output_file = os.path.join(output_dir, "disulfide_schematic.pdf")
    create_disulfide_schematic(
        disulfide=ss,
        output_file=output_file,
        show_angles=True,
        show_ca_ca_distance=True,
        style="detailed",
    )
    print(f"Saved PDF schematic to: {output_file}")


if __name__ == "__main__":
    main()
