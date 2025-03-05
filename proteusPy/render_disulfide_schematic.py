#!/usr/bin/env python3
"""
Stand-alone program for creating 2D schematic diagrams of disulfide bonds.

This program provides a command-line interface to the disulfide_schematic module
in the proteusPy package, allowing users to create publication-ready 2D diagrams
of disulfide bonds with various customization options.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-04
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt

import proteusPy as pp
from proteusPy.disulfide_schematic import (
    create_disulfide_schematic,
    create_disulfide_schematic_from_model,
)

# Add the parent directory to the path so we can import proteusPy
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


PDB_SS = None
best_id = "2q7q_75D_140D"
worst_id = "6vxk_801B_806B"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the disulfide schematic renderer.

    :return: Parsed command-line arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Create 2D schematic diagrams of disulfide bonds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdb_id",
        type=str,
        help="PDB ID to load disulfides from (will use the first disulfide found)",
    )
    input_group.add_argument(
        "--ss_id",
        type=str,
        default=best_id,
        help="Specific disulfide ID to render (format: pdbid_resnum1chain_resnum2chain)",
    )
    input_group.add_argument(
        "--model",
        action="store_true",
        help="Create a model disulfide with specified dihedral angles",
    )

    # Model disulfide parameters
    model_group = parser.add_argument_group("Model disulfide parameters")
    model_group.add_argument(
        "--chi1",
        type=float,
        default=-60.0,
        help="Chi1 dihedral angle for model disulfide",
    )
    model_group.add_argument(
        "--chi2",
        type=float,
        default=-60.0,
        help="Chi2 dihedral angle for model disulfide",
    )
    model_group.add_argument(
        "--chi3",
        type=float,
        default=-90.0,
        help="Chi3 dihedral angle for model disulfide",
    )
    model_group.add_argument(
        "--chi4",
        type=float,
        default=-60.0,
        help="Chi4 dihedral angle for model disulfide",
    )
    model_group.add_argument(
        "--chi5",
        type=float,
        default=-60.0,
        help="Chi5 dihedral angle for model disulfide",
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument(
        "--style",
        type=str,
        choices=["publication", "simple", "detailed"],
        default="publication",
        help="Visualization style",
    )
    viz_group.add_argument(
        "--show_angles",
        action="store_true",
        default=True,
        help="Show dihedral angles in the schematic",
    )
    viz_group.add_argument(
        "--show_ca_ca_distance",
        action="store_true",
        default=True,
        help="Show Cα-Cα distance in the schematic",
    )
    viz_group.add_argument(
        "--show_labels",
        action="store_true",
        default=True,
        help="Show atom labels in the schematic",
    )
    viz_group.add_argument(
        "--hide_title",
        action="store_true",
        help="Hide the title in the schematic",
    )
    viz_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster outputs (DPI)",
    )
    viz_group.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8, 6),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (width height)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output_file",
        type=str,
        help="Path to save the output file (supports .svg, .pdf, .png)",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="schematic_outputs",
        help="Directory to save the output file",
    )
    output_group.add_argument(
        "--display",
        action="store_true",
        help="Display the schematic instead of saving to a file",
    )

    # Database options
    db_group = parser.add_argument_group("Database options")
    db_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output during database loading",
    )

    return parser.parse_args()


def load_disulfide(
    args: argparse.Namespace,
) -> Optional[pp.DisulfideBase.Disulfide]:
    """
    Load a disulfide based on the provided command-line arguments.

    :param args: Parsed command-line arguments
    :type args: argparse.Namespace
    :return: Loaded disulfide object or None if not found
    :rtype: Optional[pp.DisulfideBase.Disulfide]
    """
    global PDB_SS

    PDB_SS = pp.Load_PDB_SS(verbose=args.verbose, subset=False)

    if args.model:
        # Create a model disulfide with specified dihedral angles
        model_ss = pp.DisulfideBase.Disulfide("model")
        model_ss.build_model(args.chi1, args.chi2, args.chi3, args.chi4, args.chi5)
        return model_ss

    if args.pdb_id:
        # Load disulfides from a specific PDB ID
        pdb_ss = PDB_SS[args.pdb_id]

        if len(pdb_ss) == 0:
            print(f"No disulfides found for PDB ID: {args.pdb_id}")
            return None
        return pdb_ss[0]  # Return the first disulfide

    if args.ss_id:
        return PDB_SS[args.ss_id]

    return None


def get_output_filename(
    args: argparse.Namespace, disulfide: pp.DisulfideBase.Disulfide
) -> Optional[str]:
    """
    Determine the output filename based on command-line arguments and disulfide properties.

    :param args: Parsed command-line arguments
    :type args: argparse.Namespace
    :param disulfide: Disulfide object to render
    :type disulfide: pp.DisulfideBase.Disulfide
    :return: Output filename or None if display only
    :rtype: Optional[str]
    """
    if args.display:
        return None

    if args.output_file:
        # Use the specified output file
        if os.path.isabs(args.output_file):
            return args.output_file
        else:
            return os.path.join(args.output_dir, args.output_file)

    # Create a default filename based on the disulfide properties
    if disulfide.name == "model":
        base_name = f"model_chi1_{args.chi1}_chi2_{args.chi2}_chi3_{args.chi3}_chi4_{args.chi4}_chi5_{args.chi5}"
    else:
        base_name = f"{disulfide.pdb_id}_{disulfide.proximal}{disulfide.proximal_chain}_{disulfide.distal}{disulfide.distal_chain}"

    filename = f"{base_name}_{args.style}.png"
    return os.path.join(args.output_dir, filename)


def render_disulfide_schematic(
    disulfide: pp.DisulfideBase.Disulfide, args: argparse.Namespace
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Render a disulfide schematic based on the provided disulfide and arguments.

    :param disulfide: Disulfide object to render
    :type disulfide: pp.DisulfideBase.Disulfide
    :param args: Parsed command-line arguments
    :type args: argparse.Namespace
    :return: Matplotlib figure and axes objects
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    output_file = get_output_filename(args, disulfide)

    # Create output directory if it doesn't exist and we're saving to a file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Render the schematic
    if disulfide.name == "model" and args.model:
        # For model disulfides, use the create_disulfide_schematic_from_model function
        fig, ax = create_disulfide_schematic_from_model(
            chi1=args.chi1,
            chi2=args.chi2,
            chi3=args.chi3,
            chi4=args.chi4,
            chi5=args.chi5,
            output_file=output_file,
            show_labels=args.show_labels,
            show_angles=args.show_angles,
            show_ca_ca_distance=args.show_ca_ca_distance,
            style=args.style,
            dpi=args.dpi,
            figsize=args.figsize,
        )
    else:
        # For real disulfides, use the create_disulfide_schematic function
        # Check if the hide_title argument was provided
        show_title = not args.hide_title if hasattr(args, "hide_title") else True

        fig, ax = create_disulfide_schematic(
            disulfide=disulfide,
            output_file=output_file,
            show_labels=args.show_labels,
            show_angles=args.show_angles,
            show_title=show_title,
            show_ca_ca_distance=args.show_ca_ca_distance,
            style=args.style,
            dpi=args.dpi,
            figsize=args.figsize,
        )

    return fig, ax


def main():
    """
    Main function for the disulfide schematic renderer.

    This function parses command-line arguments, loads the specified disulfide,
    renders the schematic, and either saves it to a file or displays it.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load the disulfide
    disulfide = load_disulfide(args)
    if disulfide is None:
        sys.exit(1)

    # Print information about the disulfide
    if disulfide.name == "model":
        print(
            f"Rendering model disulfide with angles: "
            f"χ₁={args.chi1:.1f}°, χ₂={args.chi2:.1f}°, χ₃={args.chi3:.1f}°, "
            f"χ₄={args.chi4:.1f}°, χ₅={args.chi5:.1f}°"
        )
    else:
        print(
            f"Rendering disulfide: {disulfide.pdb_id} "
            f"{disulfide.proximal}{disulfide.proximal_chain}-"
            f"{disulfide.distal}{disulfide.distal_chain}"
        )
        print(
            f"Energy: {disulfide.energy:.2f} kcal/mol, "
            f"Torsion Length: {disulfide.torsion_length:.2f}°, "
            f"Cα Distance: {disulfide.ca_distance:.2f} Å"
        )

    # Render the schematic
    fig, ax = render_disulfide_schematic(disulfide, args)

    # Display the schematic if requested
    if args.display:
        print("Displaying schematic...")
        plt.show()
    else:
        output_file = get_output_filename(args, disulfide)
        print(f"Saved schematic to: {output_file}")


if __name__ == "__main__":
    main()
