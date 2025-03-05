"""
This module provides functionality to create 2D schematic diagrams of disulfide bonds
for publication purposes.

Author: Eric G. Suchanek, PhD
Last revision: 2025-03-04 10:15:04
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.path import Path

from proteusPy.atoms import ATOM_COLORS
from proteusPy.DisulfideBase import Disulfide


def create_disulfide_schematic(
    disulfide=None,
    output_file=None,
    show_labels=True,
    show_angles=False,
    show_title=True,
    show_ca_ca_distance=False,
    style="publication",
    dpi=300,
    figsize=(8, 6),
):
    """
    Create a 2D schematic diagram of a disulfide bond for publication.

    :param disulfide: The disulfide bond object to visualize. If None, creates a model disulfide.
    :type disulfide: Disulfide, optional
    :param output_file: Path to save the output file (supports .svg, .pdf, .png). If None, displays the figure.
    :type output_file: str, optional
    :param show_labels: Whether to show atom labels
    :type show_labels: bool, default=True
    :param show_angles: Whether to show dihedral angles
    :type show_angles: bool, default=False
    :param show_title: Whether to show the title with disulfide information
    :type show_title: bool, default=True
    :param show_ca_ca_distance: Whether to show a line indicating the Cα-Cα distance
    :type show_ca_ca_distance: bool, default=False
    :param style: Visualization style ("publication", "simple", "detailed")
    :type style: str, default="publication"
    :param dpi: Resolution for raster outputs
    :type dpi: int, default=300
    :param figsize: Figure size in inches
    :type figsize: tuple, default=(8, 6)
    :return: Matplotlib figure and axis objects
    :rtype: tuple
    """
    # Import networkx inside the function to ensure it's available
    import networkx as nx

    # Create a model disulfide if none provided
    if disulfide is None:
        disulfide = Disulfide("model")
        disulfide.build_model(-60, -60, -90, -60, -60)  # Standard conformation

    # Create a graph
    G = nx.Graph()

    # Define atom positions (2D layout)
    # These positions create a clear 2D representation of the disulfide bond
    positions = {
        # Proximal residue
        "N_prox": (0, 0),
        "CA_prox": (1, 0),
        "C_prox": (2, 0),
        "O_prox": (2, 1),
        "CB_prox": (1, -1),
        "SG_prox": (1, -2),
        # Distal residue
        "N_dist": (5, 0),
        "CA_dist": (4, 0),
        "C_dist": (3, 0),
        "O_dist": (3, 1),
        "CB_dist": (4, -1),
        "SG_dist": (4, -2),
    }

    # Define atom properties
    atom_types = {
        "N_prox": "N",
        "CA_prox": "C",
        "C_prox": "C",
        "O_prox": "O",
        "CB_prox": "C",
        "SG_prox": "SG",
        "N_dist": "N",
        "CA_dist": "C",
        "C_dist": "C",
        "O_dist": "O",
        "CB_dist": "C",
        "SG_dist": "SG",
    }

    # Define atom labels (can be different from node names)
    atom_labels = {
        "N_prox": "N",
        "CA_prox": "Cα",
        "C_prox": "C'",
        "O_prox": "O",
        "CB_prox": "Cβ",
        "SG_prox": "Sγ",
        "N_dist": "N",
        "CA_dist": "Cα",
        "C_dist": "C'",
        "O_dist": "O",
        "CB_dist": "Cβ",
        "SG_dist": "Sγ",
    }

    # Add nodes to the graph
    for node, pos in positions.items():
        atom_type = atom_types[node]
        G.add_node(node, pos=pos, atom_type=atom_type, label=atom_labels[node])

    # Define bonds
    bonds = [
        # Proximal residue
        ("N_prox", "CA_prox"),
        ("CA_prox", "C_prox"),
        ("C_prox", "O_prox"),
        ("CA_prox", "CB_prox"),
        ("CB_prox", "SG_prox"),
        # Distal residue
        ("N_dist", "CA_dist"),
        ("CA_dist", "C_dist"),
        ("C_dist", "O_dist"),
        ("CA_dist", "CB_dist"),
        ("CB_dist", "SG_dist"),
        # Disulfide bond
        ("SG_prox", "SG_dist"),
    ]

    # Add edges to the graph
    for bond in bonds:
        G.add_edge(bond[0], bond[1])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set style parameters based on selected style
    if style == "simple":
        node_size = 500
        font_size = 10
        edge_width = 1.5
        ss_edge_width = 2.0
    elif style == "detailed":
        node_size = 700
        font_size = 12
        edge_width = 2.0
        ss_edge_width = 3.0
    else:  # "publication" (default)
        node_size = 600
        font_size = 11
        edge_width = 1.8
        ss_edge_width = 2.5

    # Draw the graph
    pos = nx.get_node_attributes(G, "pos")

    # Draw edges (bonds)
    for edge in G.edges():
        if (edge[0] == "SG_prox" and edge[1] == "SG_dist") or (
            edge[0] == "SG_dist" and edge[1] == "SG_prox"
        ):
            # Draw disulfide bond (double line)
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            # Calculate perpendicular offset for double line
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            offset = 0.05  # Offset for double line

            # Normalize and rotate by 90 degrees for perpendicular offset
            nx, ny = -dy / length, dx / length

            # Draw two parallel lines (one solid, one dashed to indicate partial double bond character)
            ax.plot(
                [x0 + offset * nx, x1 + offset * nx],
                [y0 + offset * ny, y1 + offset * ny],
                color="gold",
                linewidth=ss_edge_width,
                solid_capstyle="round",
            )
            ax.plot(
                [x0 - offset * nx, x1 - offset * nx],
                [y0 - offset * ny, y1 - offset * ny],
                color="gold",
                linewidth=ss_edge_width,
                linestyle="dashed",  # Dashed line for partial double bond character
                dash_capstyle="round",
            )
        elif (
            (edge[0] == "C_prox" and edge[1] == "O_prox")
            or (edge[0] == "O_prox" and edge[1] == "C_prox")
            or (edge[0] == "C_dist" and edge[1] == "O_dist")
            or (edge[0] == "O_dist" and edge[1] == "C_dist")
        ):
            # Draw C-O bond (double line in black)
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            # Calculate perpendicular offset for double line
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            offset = 0.05  # Offset for double line

            # Normalize and rotate by 90 degrees for perpendicular offset
            nx, ny = -dy / length, dx / length

            # Draw two parallel lines in black
            ax.plot(
                [x0 + offset * nx, x1 + offset * nx],
                [y0 + offset * ny, y1 + offset * ny],
                color="black",
                linewidth=edge_width,
                solid_capstyle="round",
            )
            ax.plot(
                [x0 - offset * nx, x1 - offset * nx],
                [y0 - offset * ny, y1 - offset * ny],
                color="black",
                linewidth=edge_width,
                solid_capstyle="round",
            )
        else:
            # Draw regular bond
            ax.plot(
                [pos[edge[0]][0], pos[edge[1]][0]],
                [pos[edge[0]][1], pos[edge[1]][1]],
                color="black",
                linewidth=edge_width,
                solid_capstyle="round",
            )

    # Draw nodes (atoms)
    for node in G.nodes():
        atom_type = G.nodes[node]["atom_type"]
        color = ATOM_COLORS.get(atom_type, "grey")

        # Convert color name to RGBA
        rgba_color = to_rgba(color)

        # Draw the atom
        circle = plt.Circle(
            pos[node],
            radius=node_size / 8000,
            facecolor=rgba_color,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(circle)

        # Add atom labels if requested
        if show_labels:
            label_offset_x = 0
            label_offset_y = 0

            # Adjust label positions for specific atoms
            if node in ["O_prox", "O_dist"]:
                label_offset_y = 0.15
            elif node in ["N_prox", "N_dist"]:
                label_offset_x = -0.15

            ax.text(
                pos[node][0] + label_offset_x,
                pos[node][1] + label_offset_y,
                G.nodes[node]["label"],
                fontsize=font_size,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

    # Add dihedral angle labels if requested
    if show_angles:
        # Chi1
        ax.text(
            1.2, -0.5, f"χ₁: {disulfide.chi1:.1f}°", fontsize=font_size - 1, ha="left"
        )
        # Chi2
        ax.text(
            1.2, -1.5, f"χ₂: {disulfide.chi2:.1f}°", fontsize=font_size - 1, ha="left"
        )
        # Chi3
        ax.text(
            2.5, -2.2, f"χ₃: {disulfide.chi3:.1f}°", fontsize=font_size - 1, ha="center"
        )
        # Chi4
        ax.text(
            3.8, -1.5, f"χ₂′: {disulfide.chi4:.1f}°", fontsize=font_size - 1, ha="right"
        )
        # Chi5
        ax.text(
            3.8, -0.5, f"χ₁′: {disulfide.chi5:.1f}°", fontsize=font_size - 1, ha="right"
        )

    # Add title if requested
    if show_title and disulfide:
        title = f"Disulfide Bond: {disulfide.pdb_id} {disulfide.proximal}{disulfide.proximal_chain}-{disulfide.distal}{disulfide.distal_chain}"
        if show_angles:
            title += f"\nEnergy: {disulfide.energy:.2f} kcal/mol, Torsion Length: {disulfide.torsion_length:.2f}°"
        ax.set_title(title, fontsize=font_size + 2)

    # Add residue labels with actual residue information if available
    proximal_label = "Proximal Cysteine"
    distal_label = "Distal Cysteine"

    # If we have a real disulfide (not a model), use the actual residue information
    if disulfide and disulfide.name != "model":
        proximal_label = f"Cys {disulfide.proximal}{disulfide.proximal_chain}"
        distal_label = f"Cys {disulfide.distal}{disulfide.distal_chain}"

    ax.text(
        1,
        1,
        proximal_label,
        fontsize=font_size + 1,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", pad=5),
    )
    ax.text(
        4,
        1,
        distal_label,
        fontsize=font_size + 1,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", pad=5),
    )

    # Add Ca-Ca distance line if requested
    if show_ca_ca_distance and disulfide:
        # Get positions of CA atoms
        ca_prox_pos = pos["CA_prox"]
        ca_dist_pos = pos["CA_dist"]

        # Create a curved path for the Ca-Ca distance line to go beneath the C' atoms
        # Calculate control points for the curved line
        # We'll use a quadratic Bezier curve that dips below the straight line
        mid_x = (ca_prox_pos[0] + ca_dist_pos[0]) / 2
        mid_y = (
            ca_prox_pos[1] + ca_dist_pos[1]
        ) / 2 - 0.5  # Offset below the straight line

        # Create the curved path

        verts = [
            ca_prox_pos,  # Start point (CA_prox)
            (mid_x, mid_y),  # Control point
            ca_dist_pos,  # End point (CA_dist)
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor="purple",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=1,  # Ensure it's drawn below other elements
        )
        ax.add_patch(patch)

        # Add distance label at the bottom of the curve
        ax.text(
            mid_x,
            mid_y + 0.05,  # Position below the curve
            f"{disulfide.ca_distance:.2f} Å",
            fontsize=font_size,
            ha="center",
            va="center",
            color="purple",
            zorder=2,  # Ensure it's drawn above the line
        )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")

    # Adjust limits to ensure all elements are visible
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2.5, 1.5)

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close()

    return fig, ax


def create_disulfide_schematic_from_model(
    chi1=-60,
    chi2=-60,
    chi3=-90,
    chi4=-60,
    chi5=-60,
    output_file=None,
    show_labels=True,
    show_angles=True,
    show_ca_ca_distance=False,
    style="publication",
    dpi=300,
    figsize=(8, 6),
):
    """
    Create a 2D schematic diagram of a model disulfide bond with specified dihedral angles.

    :param chi1: Dihedral angle chi1 for the disulfide bond
    :type chi1: float, default=-60
    :param chi2: Dihedral angle chi2 for the disulfide bond
    :type chi2: float, default=-60
    :param chi3: Dihedral angle chi3 for the disulfide bond
    :type chi3: float, default=-90
    :param chi4: Dihedral angle chi4 for the disulfide bond
    :type chi4: float, default=-60
    :param chi5: Dihedral angle chi5 for the disulfide bond
    :type chi5: float, default=-60
    :param output_file: Path to save the output file (supports .svg, .pdf, .png). If None, displays the figure.
    :type output_file: str, optional
    :param show_labels: Whether to show atom labels
    :type show_labels: bool, default=True
    :param show_angles: Whether to show dihedral angles
    :type show_angles: bool, default=True
    :param show_ca_ca_distance: Whether to show a line indicating the Cα-Cα distance
    :type show_ca_ca_distance: bool, default=False
    :param style: Visualization style ("publication", "simple", "detailed")
    :type style: str, default="publication"
    :param dpi: Resolution for raster outputs
    :type dpi: int, default=300
    :param figsize: Figure size in inches
    :type figsize: tuple, default=(8, 6)
    :return: Matplotlib figure and axis objects
    :rtype: tuple
    """
    # Create a model disulfide with the specified angles
    model_ss = Disulfide("model")
    model_ss.build_model(chi1, chi2, chi3, chi4, chi5)

    # Create the schematic
    return create_disulfide_schematic(
        disulfide=model_ss,
        output_file=output_file,
        show_labels=show_labels,
        show_angles=show_angles,
        show_ca_ca_distance=show_ca_ca_distance,
        style=style,
        dpi=dpi,
        figsize=figsize,
    )


if __name__ == "__main__":
    # Example usage
    import proteusPy as pp

    # Load a disulfide from the database
    PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
    ss = PDB_SS[0]

    # Create and save a schematic
    fig, ax = create_disulfide_schematic(
        disulfide=ss,
        output_file="disulfide_schematic.png",
        show_angles=True,
        style="publication",
    )

    # Create a model disulfide schematic
    fig2, ax2 = create_disulfide_schematic_from_model(
        chi1=-60,
        chi2=-60,
        chi3=-90,
        chi4=-60,
        chi5=-60,
        output_file="model_disulfide_schematic.png",
        show_angles=True,
    )

    plt.show()

# End of disulfide_schematic.py
