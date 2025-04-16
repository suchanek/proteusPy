"""
Example demonstrating the DisulfideEnergy class for analyzing energy components
of disulfide bonds with various conformations.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from proteusPy.DisulfideBase import Disulfide
from proteusPy.DisulfideEnergy import DisulfideEnergy


def compare_energy_classes():
    """Compare the energy values from DisulfideBase and DisulfideEnergy classes"""
    print(
        "Comparing energy calculations between DisulfideBase and DisulfideEnergy classes"
    )
    print("-" * 80)

    # Define some common disulfide conformations
    conformations = {
        "Spiral": [-60, -60, -85, -60, -60],
        "Hook": [-60, -60, -85, 60, 60],
        "Staple": [60, 60, 85, 60, 60],
    }

    # Compare energy calculations for each conformation
    for name, dihedrals in conformations.items():
        print(f"\nConformation: {name} {dihedrals}")

        # Calculate energy using DisulfideBase
        ss = Disulfide("test")
        ss.dihedrals = dihedrals

        # Calculate energy using DisulfideEnergy
        ss_energy = DisulfideEnergy(*dihedrals)

        # Compare results
        print(
            f"  DisulfideBase:   Standard Energy = {ss.energy:.4f} kcal/mol, DSE Energy = {ss.TorsionEnergyKJ:.4f} kJ/mol"
        )
        print(
            f"  DisulfideEnergy: Standard Energy = {ss_energy.standard_energy:.4f} kcal/mol, DSE Energy = {ss_energy.dse_energy:.4f} kJ/mol"
        )

        # Display detailed component breakdown
        print("\nDetailed energy component breakdown:")
        print(ss_energy.summary())
        print("-" * 80)


def demonstrate_energy_plot():
    """Demonstrate the plot_energy_components_scan method of DisulfideEnergy class"""
    print("Creating energy component scan plot...")

    # Create DisulfideEnergy object with a spiral conformation as starting point
    energy = DisulfideEnergy(-60, -60, -85, -60, -60)

    # Plot energy components varying chi3
    energy.plot_energy_components_scan(
        angle_to_vary=3,
        filename="energy_components_scan_chi3.png",
        colorblind_friendly=True,
    )

    # Plot energy components varying chi1 with different fixed angles
    energy.set_dihedrals(60, 60, 85, 60, 60)  # Change to staple conformation
    energy.plot_energy_components_scan(
        angle_to_vary=1,
        filename="energy_components_scan_chi1.png",
        colorblind_friendly=True,
    )


def create_surface_plot(chi_indices=(2, 3)):
    """
    Create a surface plot showing energy variation with two chi angles

    Parameters
    ----------
    chi_indices : tuple
        Indices of the two chi angles to vary (1-based, like chi1, chi2, etc.)
    """
    print(f"Creating surface plot for chi{chi_indices[0]} and chi{chi_indices[1]}...")

    # Set default values for all chi angles
    default_chi = [-60, -60, -85, -60, -60]  # Spiral conformation

    # Create a grid of values for the two selected chi angles
    angle_range = np.linspace(-180, 180, 60)
    x, y = np.meshgrid(angle_range, angle_range)

    # Arrays to store energy values
    z_standard = np.zeros_like(x)
    z_dse = np.zeros_like(x)

    # Calculate energy for each combination of angles
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Create a copy of the default chi values
            chi_values = default_chi.copy()

            # Update the chi values being varied
            chi_values[chi_indices[0] - 1] = x[i, j]
            chi_values[chi_indices[1] - 1] = y[i, j]

            # Calculate energy
            energy = DisulfideEnergy(*chi_values)
            z_standard[i, j] = energy.standard_energy
            z_dse[i, j] = energy.dse_energy_kcal

    # Create the surface plot
    fig = plt.figure(figsize=(18, 8))

    # Standard energy surface
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        x, y, z_standard, cmap="viridis", linewidth=0, antialiased=True, alpha=0.8
    )

    ax1.set_title(
        f"Standard Energy (kcal/mol) - chi{chi_indices[0]} vs chi{chi_indices[1]}"
    )
    ax1.set_xlabel(f"Chi{chi_indices[0]} (degrees)")
    ax1.set_ylabel(f"Chi{chi_indices[1]} (degrees)")
    ax1.set_zlabel("Energy (kcal/mol)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # DSE energy surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        x, y, z_dse, cmap="plasma", linewidth=0, antialiased=True, alpha=0.8
    )

    ax2.set_title(f"DSE Energy (kcal/mol) - chi{chi_indices[0]} vs chi{chi_indices[1]}")
    ax2.set_xlabel(f"Chi{chi_indices[0]} (degrees)")
    ax2.set_ylabel(f"Chi{chi_indices[1]} (degrees)")
    ax2.set_zlabel("Energy (kcal/mol)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.savefig(
        f"energy_surface_chi{chi_indices[0]}_chi{chi_indices[1]}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    # Run demonstrations
    compare_energy_classes()
    demonstrate_energy_plot()  # Use the class method instead of the standalone function
    create_surface_plot(chi_indices=(2, 3))  # Vary chi2 and chi3
