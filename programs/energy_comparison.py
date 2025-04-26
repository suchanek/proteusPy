import matplotlib.pyplot as plt
import numpy as np

from proteusPy import Disulfide
from proteusPy.DisulfideEnergy import DisulfideEnergy

# Create angle range for visualization
angles = np.linspace(-180, 180, 360)

# Calculate energy components across angle range for each chi angle
# We'll vary one angle at a time while keeping others at 0
components_by_chi = {}

for chi_to_vary in range(1, 6):
    components_by_chi[f"chi{chi_to_vary}"] = []
    for angle in angles:
        # Set up the chi angles
        chi1, chi2, chi3, chi4, chi5 = [0.0] * 5
        if chi_to_vary == 1:
            chi1 = angle
        elif chi_to_vary == 2:
            chi2 = angle
        elif chi_to_vary == 3:
            chi3 = angle
        elif chi_to_vary == 4:
            chi4 = angle
        else:  # chi_to_vary == 5
            chi5 = angle

        # Get components
        energy = DisulfideEnergy(chi1, chi2, chi3, chi4, chi5)
        std_components = energy.calculate_standard_components()
        dse_components = energy.calculate_dse_components()
        components_by_chi[f"chi{chi_to_vary}"].append((std_components, dse_components))

# Create figure with two rows for each energy model
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Color map for different components
colors = ["blue", "red", "green", "orange", "purple"]
line_styles = ["-", "--", ":", "-."]

# Plot standard energy components
ax1 = axs[0]
for i, component_name in enumerate(["chi1_chi5", "chi2_chi4", "chi3", "constant"]):
    for chi_idx, chi_name in enumerate(
        ["chi1", "chi3", "chi5"]
    ):  # Select a few key chi angles to avoid clutter
        if chi_idx > 0 and component_name == "constant":
            continue  # Skip constant for duplicate plots

        values = [
            components_by_chi[chi_name][j][0][component_name]
            for j in range(len(angles))
        ]
        ax1.plot(
            angles,
            values,
            label=f"{component_name} ({chi_name} varied)",
            color=colors[i % len(colors)],
            linestyle=line_styles[chi_idx % len(line_styles)],
        )

ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax1.set_title("Standard Energy Components (kcal/mol)")
ax1.set_xlabel("Angle (degrees)")
ax1.set_ylabel("Energy (kcal/mol)")
ax1.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
ax1.grid(True, alpha=0.3)

# Plot DSE components
ax2 = axs[1]
for i, component_name in enumerate(
    ["chi1_chi5", "chi2_chi4", "chi3_2fold", "chi3_3fold"]
):
    for chi_idx, chi_name in enumerate(
        ["chi1", "chi3", "chi5"]
    ):  # Select a few key chi angles to avoid clutter
        values = [
            components_by_chi[chi_name][j][1][component_name]
            for j in range(len(angles))
        ]
        ax2.plot(
            angles,
            values,
            label=f"{component_name} ({chi_name} varied)",
            color=colors[i % len(colors)],
            linestyle=line_styles[chi_idx % len(line_styles)],
        )

ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax2.set_title("DSE Components (kJ/mol)")
ax2.set_xlabel("Angle (degrees)")
ax2.set_ylabel("Energy (kJ/mol)")
ax2.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
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
            # Use Disulfide for standard energy
            ss = Disulfide("test")
            ss.dihedrals = [
                chi1,
                chi2_vals[i, j, k],
                chi3_vals[i, j, k],
                chi4_vals[i, j, k],
                chi5,
            ]
            energies_kcal[i, j, k] = ss.TorsionEnergy
            
            # Use DisulfideEnergy for DSE energy
            ss_energy = DisulfideEnergy(
                chi1,
                chi2_vals[i, j, k],
                chi3_vals[i, j, k],
                chi4_vals[i, j, k],
                chi5
            )
            energies_kj[i, j, k] = ss_energy.dse_energy

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
print("\nDSE Function (kJ/mol):")
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
