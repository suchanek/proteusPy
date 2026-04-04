#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for comparing disulfides from different structural classes.

This script demonstrates how to:
1. Generate disulfides for multiple classes
2. Compare the energy distributions between classes
3. Compare the dihedral angle distributions between classes
4. Identify the minimum energy disulfide for each class
5. Visualize the minimum energy disulfides for comparison

Author: Eric G. Suchanek, PhD
Last Modification: 2025-04-27
"""

import os
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from proteusPy import Disulfide, DisulfideList
from proteusPy.DisulfideClassGenerator import DisulfideClassGenerator

# Add a global save directory constant and create it
SAVE_DIR = Path("class_analysis_outputs")
SAVE_DIR.mkdir(exist_ok=True)


def compare_energy_distributions(
    class_disulfides: Dict[str, DisulfideList],
    class_names: Dict[str, str],
    save_dir: Path = SAVE_DIR,
) -> None:
    """
    Compare the energy distributions of disulfides from different classes.

    Parameters:
        class_disulfides (Dict[str, DisulfideList]): Dictionary mapping class IDs to DisulfideLists.
        class_names (Dict[str, str]): Dictionary mapping class IDs to class names.
    """
    plt.figure(figsize=(12, 8))

    # Define a list of colors for different classes
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Create a histogram for each class
    for i, (class_id, disulfide_list) in enumerate(class_disulfides.items()):
        # Extract energies from all disulfides
        energies = [ss.energy for ss in disulfide_list]

        # Calculate statistics
        min_energy = min(energies)
        mean_energy = np.mean(energies)

        # Plot histogram
        color = colors[i % len(colors)]
        plt.hist(
            energies,
            bins=20,
            alpha=0.3,
            color=color,
            label=f"Class {class_id} ({class_names[class_id]})",
        )

        # Plot vertical lines for mean and min energy
        plt.axvline(
            mean_energy,
            color=color,
            linestyle="dashed",
            linewidth=2,
            label=f"Mean {class_id}: {mean_energy:.2f}",
        )
        plt.axvline(
            min_energy,
            color=color,
            linestyle="dotted",
            linewidth=2,
            label=f"Min {class_id}: {min_energy:.2f}",
        )

    plt.title("Energy Distribution Comparison Between Classes")
    plt.xlabel("Energy (kcal/mol)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    outpath = save_dir / "class_energy_comparison.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved energy distribution comparison plot to {outpath}")

    # Show the plot (comment out if running in a non-interactive environment)
    # plt.show()


def compare_dihedral_distributions(
    class_disulfides: Dict[str, DisulfideList],
    class_names: Dict[str, str],
    save_dir: Path = SAVE_DIR,
) -> None:
    """
    Compare the dihedral angle distributions of disulfides from different classes.

    Parameters:
        class_disulfides (Dict[str, DisulfideList]): Dictionary mapping class IDs to DisulfideLists.
        class_names (Dict[str, str]): Dictionary mapping class IDs to class names.
    """
    # Create a figure with subplots for each dihedral angle
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    # Define a list of colors for different classes
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Plot histograms for each dihedral angle and each class
    for i, (class_id, disulfide_list) in enumerate(class_disulfides.items()):
        # Extract dihedral angles from all disulfides
        chi1_values = [ss.chi1 for ss in disulfide_list]
        chi2_values = [ss.chi2 for ss in disulfide_list]
        chi3_values = [ss.chi3 for ss in disulfide_list]
        chi4_values = [ss.chi4 for ss in disulfide_list]
        chi5_values = [ss.chi5 for ss in disulfide_list]

        color = colors[i % len(colors)]

        # Plot histograms for each dihedral angle
        axs[0].hist(
            chi1_values,
            bins=20,
            alpha=0.3,
            color=color,
            label=f"Class {class_id} ({class_names[class_id]})",
        )
        axs[1].hist(chi2_values, bins=20, alpha=0.3, color=color)
        axs[2].hist(chi3_values, bins=20, alpha=0.3, color=color)
        axs[3].hist(chi4_values, bins=20, alpha=0.3, color=color)
        axs[4].hist(chi5_values, bins=20, alpha=0.3, color=color)

    # Set titles and labels
    axs[0].set_title("Chi1 Distribution")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].set_title("Chi2 Distribution")
    axs[1].grid(True, alpha=0.3)

    axs[2].set_title("Chi3 Distribution")
    axs[2].grid(True, alpha=0.3)

    axs[3].set_title("Chi4 Distribution")
    axs[3].grid(True, alpha=0.3)

    axs[4].set_title("Chi5 Distribution")
    axs[4].set_xlabel("Angle (degrees)")
    axs[4].grid(True, alpha=0.3)

    # Add a common y-label
    fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical", fontsize=12)

    # Add a main title
    fig.suptitle("Dihedral Angle Distribution Comparison Between Classes", fontsize=16)
    plt.tight_layout(rect=[0.05, 0, 1, 0.97])

    # Save the plot
    outpath = save_dir / "class_dihedral_comparison.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved dihedral distribution comparison plot to {outpath}")

    # Show the plot (comment out if running in a non-interactive environment)
    # plt.show()


def compare_minimum_energy_disulfides(
    class_disulfides: Dict[str, DisulfideList],
    class_names: Dict[str, str],
    save_dir: Path = SAVE_DIR,
) -> Dict[str, Disulfide]:
    """
    Compare the minimum energy disulfides from different classes.

    Parameters:
        class_disulfides (Dict[str, DisulfideList]): Dictionary mapping class IDs to DisulfideLists.
        class_names (Dict[str, str]): Dictionary mapping class IDs to class names.

    Returns:
        Dict[str, Disulfide]: Dictionary mapping class IDs to minimum energy disulfides.
    """
    min_energy_disulfides = {}

    print("\nMinimum Energy Disulfides Comparison:")
    print("-" * 50)
    print(
        f"{'Class ID':<10} {'Class Name':<15} {'Energy (kcal/mol)':<20} {'Cα Distance (Å)':<15}"
    )
    print("-" * 50)

    for class_id, disulfide_list in class_disulfides.items():
        # Find the disulfide with the minimum energy
        min_energy_disulfide = min(disulfide_list, key=lambda ss: ss.energy)
        min_energy_disulfides[class_id] = min_energy_disulfide

        # Print information about the minimum energy disulfide
        print(
            f"{class_id:<10} {class_names[class_id]:<15} {min_energy_disulfide.energy:<20.2f} {min_energy_disulfide.ca_distance:<15.2f}"
        )

        # Save the minimum energy disulfide to a file
        min_energy_file = os.path.join(
            save_dir, f"class_{class_id}_min_energy_disulfide.pkl"
        )
        with open(min_energy_file, "wb") as f:
            pickle.dump(min_energy_disulfide, f)

    print("-" * 50)

    # Create a bar chart comparing the minimum energies
    plt.figure(figsize=(10, 6))

    class_ids = list(min_energy_disulfides.keys())
    energies = [min_energy_disulfides[class_id].energy for class_id in class_ids]

    # Define colors based on energy values (lower energy = greener, higher energy = redder)
    norm = plt.Normalize(min(energies), max(energies))
    cmap = plt.cm.get_cmap("RdYlGn")
    colors = cmap(norm(energies))
    cmap = plt.cm.get_cmap("viridis")  # Replace "RdYlGn" with "viridis"
    bars = plt.bar(class_ids, energies, color=colors)

    plt.title("Minimum Energy Comparison Between Classes")
    plt.xlabel("Class ID")
    plt.ylabel("Energy (kcal/mol)")
    plt.grid(True, alpha=0.3, axis="y")

    # Add class names as annotations
    for i, _bar in enumerate(bars):
        plt.text(
            _bar.get_x() + _bar.get_width() / 2,
            _bar.get_height() + 0.1,
            class_names[class_ids[i]],
            ha="center",
            va="bottom",
            rotation=45,
        )

    # Save the plot
    outpath = os.path.join(save_dir, "class_min_energy_comparison.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved minimum energy comparison plot to {outpath}")

    # Show the plot (comment out if running in a non-interactive environment)
    # plt.show()

    return min_energy_disulfides


def compare_average_conformations(
    class_disulfides: Dict[str, DisulfideList],
    class_names: Dict[str, str],
    save_dir: str = SAVE_DIR,
) -> Dict[str, Disulfide]:
    """
    Compare the average conformations of disulfides from different classes.

    Parameters:
        class_disulfides (Dict[str, DisulfideList]): Dictionary mapping class IDs to DisulfideLists.
        class_names (Dict[str, str]): Dictionary mapping class IDs to class names.

    Returns:
        Dict[str, Disulfide]: Dictionary mapping class IDs to average conformation disulfides.
    """
    avg_conformation_disulfides = {}

    print("\nAverage Conformation Comparison:")
    print("-" * 80)
    print(
        f"{'Class ID':<10} {'Class Name':<15} {'Chi1':<10} {'Chi2':<10} {'Chi3':<10} {'Chi4':<10} {'Chi5':<10}"
    )
    print("-" * 80)

    for class_id, disulfide_list in class_disulfides.items():
        # Calculate the average conformation
        avg_conformation = disulfide_list.average_conformation

        # Create a disulfide with the average conformation
        avg_disulfide = Disulfide(
            name=f"{class_id}_{class_names[class_id]}_avg", torsions=avg_conformation
        )
        avg_conformation_disulfides[class_id] = avg_disulfide

        # Print information about the average conformation
        print(
            f"{class_id:<10} {class_names[class_id]:<15} {avg_conformation[0]:<10.2f} {avg_conformation[1]:<10.2f} {avg_conformation[2]:<10.2f} {avg_conformation[3]:<10.2f} {avg_conformation[4]:<10.2f}"
        )

        # Save the average conformation disulfide to a file
        avg_file = os.path.join(
            save_dir, f"class_{class_id}_avg_conformation_disulfide.pkl"
        )
        with open(avg_file, "wb") as f:
            pickle.dump(avg_disulfide, f)

    print("-" * 80)

    # Create a radar chart comparing the average conformations
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Define the angles for the radar chart (5 dihedral angles)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Define a list of colors and markers for different classes
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "h"]

    # Plot each class
    for i, (class_id, avg_disulfide) in enumerate(avg_conformation_disulfides.items()):
        # Normalize the dihedral angles to the range [0, 1]
        values = [angle / 360 + 0.5 for angle in avg_disulfide.dihedrals]
        values += values[:1]  # Close the loop

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2,
            marker=marker,
            label=f"{class_id} ({class_names[class_id]})",
        )
        ax.fill(angles, values, color=color, alpha=0.1)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Chi1", "Chi2", "Chi3", "Chi4", "Chi5"])

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title("Average Conformation Comparison Between Classes")

    # Save the plot
    outpath = os.path.join(save_dir, "class_avg_conformation_comparison.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved average conformation comparison plot to {outpath}")

    # Show the plot (comment out if running in a non-interactive environment)
    # plt.show()

    return avg_conformation_disulfides


def main():
    """
    Main function demonstrating how to compare disulfides from different classes.
    """

    # Select classes to compare
    selected_classes = [
        "+++++",  # "+++++" (RH Spiral)
        "-----",  # "-----" (LH Spiral)
        "-+---",  # "-+---" (RH Staple)
        "-+-+-",  # "-+-+-" (LH Staple)
        "+-++-",  # "+-++-" (RH Hook)
    ]

    print(f"Generating disulfides for selected classes: {selected_classes}...")

    # Create a generator instance and generate disulfides for the selected classes
    generator = DisulfideClassGenerator()
    class_disulfides = generator.generate_for_selected_classes(selected_classes)

    class_names = {}
    for class_id in selected_classes:
        class_names[class_id] = class_id

    # Print information about the generated disulfides
    print("\nGenerated Disulfides:")
    print("-" * 50)
    print(
        f"{'Class ID':<10} {'Class Name':<15} {'Count':<10} {'Avg Energy':<15} {'Avg Cα Dist':<15}"
    )
    print("-" * 50)

    for class_id, disulfide_list in class_disulfides.items():
        print(
            f"{class_id:<10} {class_names[class_id]:<15} {len(disulfide_list):<10} {disulfide_list.average_energy:<15.2f} {disulfide_list.average_ca_distance:<15.2f}"
        )

    print("-" * 50)

    # Compare the energy distributions
    compare_energy_distributions(class_disulfides, class_names)

    # Compare the dihedral angle distributions
    compare_dihedral_distributions(class_disulfides, class_names)

    # Compare the minimum energy disulfides
    compare_minimum_energy_disulfides(class_disulfides, class_names)

    # Compare the average conformations
    compare_average_conformations(class_disulfides, class_names)

    # Save all disulfides to a file
    for class_id, disulfide_list in class_disulfides.items():
        output_file = os.path.join(SAVE_DIR, f"class_{class_id}_disulfides.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(disulfide_list, f)
        print(f"Saved disulfides for class {class_id} to {output_file}.")

    print("\nComparison complete. All results have been saved to files.")


if __name__ == "__main__":
    main()
