import matplotlib.pyplot as plt
import numpy as np

from proteusPy.DisulfideBase import Disulfide


def calculate_energy_components(chi1, chi2, chi3, chi4, chi5):
    """Calculate individual components of both energy functions"""

    # Convert to radians for direct calculation
    def torad(deg):
        return np.deg2rad(deg)

    # Standard energy components (kcal/mol)
    std_components = {
        "chi1_chi5": 2.0 * (np.cos(torad(3.0 * chi1)) + np.cos(torad(3.0 * chi5))),
        "chi2_chi4": np.cos(torad(3.0 * chi2)) + np.cos(torad(3.0 * chi4)),
        "chi3": 3.5 * np.cos(torad(2.0 * chi3)) + 0.6 * np.cos(torad(3.0 * chi3)),
        "constant": 10.1,
    }

    # DSE components (kJ/mol)
    dse_components = {
        "chi1_chi5": 8.37
        * ((1 + np.cos(3 * torad(chi1))) + (1 + np.cos(3 * torad(chi5)))),
        "chi2_chi4": 4.18
        * ((1 + np.cos(3 * torad(chi2))) + (1 + np.cos(3 * torad(chi4)))),
        "chi3_2fold": 14.64 * (1 + np.cos(2 * torad(chi3))),
        "chi3_3fold": 2.51 * (1 + np.cos(3 * torad(chi3))),
    }

    return std_components, dse_components


# Create angle range for visualization
angles = np.linspace(-180, 180, 360)

# Calculate energy components across angle range
chi_test = 0  # Fix other angles at 0 for visualization
components_by_angle = {
    angle: calculate_energy_components(angle, chi_test, chi_test, chi_test, chi_test)
    for angle in angles
}

# Extract components for plotting
std_chi1_chi5 = [
    components[0]["chi1_chi5"] for components in components_by_angle.values()
]
dse_chi1_chi5 = [
    components[1]["chi1_chi5"] for components in components_by_angle.values()
]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot standard energy components
ax1.plot(angles, std_chi1_chi5, label="chi1+chi5 term", color="blue")
ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax1.set_title("Standard Energy Components (kcal/mol)")
ax1.set_xlabel("Angle (degrees)")
ax1.set_ylabel("Energy (kcal/mol)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot DSE components
ax2.plot(angles, dse_chi1_chi5, label="chi1+chi5 term", color="red")
ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax2.set_title("DSE Components (kJ/mol)")
ax2.set_xlabel("Angle (degrees)")
ax2.set_ylabel("Energy (kJ/mol)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_components.png", dpi=300, bbox_inches="tight")
plt.close()

# Now create a systematic comparison varying chi2, chi3, and chi4
chi_range = np.linspace(-180, 180, 20)
chi2_vals, chi3_vals, chi4_vals = np.meshgrid(chi_range, chi_range, chi_range)

# Fix chi1 and chi5
chi1 = chi5 = -60.0

# Arrays to store energies
energies_kcal = np.zeros_like(chi2_vals)
energies_kj = np.zeros_like(chi2_vals)

# Calculate energies for each combination
for i in range(chi2_vals.shape[0]):
    for j in range(chi2_vals.shape[1]):
        for k in range(chi2_vals.shape[2]):
            ss = Disulfide("test")
            ss.dihedrals = [
                chi1,
                chi2_vals[i, j, k],
                chi3_vals[i, j, k],
                chi4_vals[i, j, k],
                chi5,
            ]
            energies_kcal[i, j, k] = ss.energy
            energies_kj[i, j, k] = ss.TorsionEnergyKJ

# Convert standard energy to kJ/mol for comparison
energies_kcal_kj = energies_kcal * 4.184

# Create scatter plot comparing the two energy functions
plt.figure(figsize=(10, 8))
plt.scatter(energies_kcal_kj.flatten(), energies_kj.flatten(), alpha=0.1)
plt.xlabel("Standard Energy Function (kJ/mol)")
plt.ylabel("DSE Function (kJ/mol)")
plt.title("Comparison of Disulfide Energy Functions")

# Add a diagonal line for reference
min_e = min(energies_kcal_kj.min(), energies_kj.min())
max_e = max(energies_kcal_kj.max(), energies_kj.max())
plt.plot([min_e, max_e], [min_e, max_e], "r--", label="y=x")
plt.legend()

# Calculate correlation coefficient
correlation = np.corrcoef(energies_kcal_kj.flatten(), energies_kj.flatten())[0, 1]
plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=plt.gca().transAxes)

plt.savefig("energy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Print statistics
print("Standard Energy Function (kJ/mol):")
print(f"  Min: {energies_kcal_kj.min():.2f}")
print(f"  Max: {energies_kcal_kj.max():.2f}")
print(f"  Mean: {energies_kcal_kj.mean():.2f}")
print(f"\nDSE Function (kJ/mol):")
print(f"  Min: {energies_kj.min():.2f}")
print(f"  Max: {energies_kj.max():.2f}")
print(f"  Mean: {energies_kj.mean():.2f}")
print(f"\nCorrelation coefficient: {correlation:.3f}")

# Calculate the percentage of negative values in standard energy
neg_values = np.sum(energies_kcal_kj < 0)
total_values = energies_kcal_kj.size
print(
    f"\nPercentage of negative values in standard energy: {100 * neg_values/total_values:.2f}%"
)
