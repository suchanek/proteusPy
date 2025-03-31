import matplotlib.pyplot as plt
import numpy as np

from proteusPy.DisulfideBase import Disulfide


def generate_and_compare_energies(n_samples=1000):
    """
    Generate synthetic dihedral angles and compare DSE vs torsional energy.

    Args:
        n_samples (int): Number of random conformations to generate

    Returns:
        tuple: Arrays of DSE and torsional energies
    """
    # Generate random dihedral angles between -180 and 180 degrees
    chi_angles = np.random.uniform(-180, 180, (n_samples, 5))

    dse_energies = []
    torsional_energies = []

    # Calculate both energies for each conformation
    for angles in chi_angles:
        ss = Disulfide("test", torsions=angles)
        dse = ss.calculate_dse()
        torsional = ss.energy * 4.184  # Convert from kcal/mol to kJ/mol

        dse_energies.append(dse)
        torsional_energies.append(torsional)

    return np.array(dse_energies), np.array(torsional_energies)


def plot_energy_comparison(dse_energies, torsional_energies):
    """
    Create a scatter plot comparing DSE vs torsional energy.

    Args:
        dse_energies (np.array): Array of DSE values
        torsional_energies (np.array): Array of torsional energy values
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(dse_energies, torsional_energies, alpha=0.5)
    plt.xlabel("DSE Energy (kJ/mol)")
    plt.ylabel("Torsional Energy (kcal/mol)")
    plt.title("Comparison of DSE vs Torsional Energy Functions")

    # Add correlation coefficient
    correlation = np.corrcoef(dse_energies, torsional_energies)[0, 1]
    plt.text(
        0.05, 0.95, f"Correlation: {correlation:.3f}", transform=plt.gca().transAxes
    )

    # Add identity line for reference
    min_val = min(min(dse_energies), min(torsional_energies))
    max_val = max(max(dse_energies), max(torsional_energies))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Generate synthetic data and create plot
    dse_energies, torsional_energies = generate_and_compare_energies()
    plot_energy_comparison(dse_energies, torsional_energies)
