"""
Example demonstrating the DisulfideEnergy class for analyzing energy components
of disulfide bonds with various conformations.
"""

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


def demonstrate_all_angles_plot():
    """Demonstrate the plot_energy_components_all_angles method of DisulfideEnergy class"""
    print("Creating combined energy component plot for all chi angles...")

    # Create DisulfideEnergy object with a spiral conformation
    energy = DisulfideEnergy(-60, -60, -85, -60, -60)

    # Plot energy components for all chi angles
    energy.plot_energy_components_all_angles(
        filename="energy_components_all_angles.png",
        colorblind_friendly=True,
    )

    print("Combined plot created and saved as 'energy_components_all_angles.png'")


if __name__ == "__main__":
    # Run demonstrations
    compare_energy_classes()
    demonstrate_energy_plot()  # Use the class method instead of the standalone function
    demonstrate_all_angles_plot()  # Demonstrate the new combined plot function
    DisulfideEnergy.create_surface_plot(chi_indices=(2, 3))  # Vary chi2 and chi3
