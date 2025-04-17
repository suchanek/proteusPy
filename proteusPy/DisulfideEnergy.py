"""
DisulfideEnergy.py - Energy calculation and analysis for disulfide bonds

This module provides a specialized class for calculating, analyzing, and visualizing
energy components in disulfide bonds. It serves as an extension to the DisulfideBase
functionality by providing more detailed energy decomposition capabilities.

The DisulfideEnergy class calculates two different energy models:
1. Standard Energy (kcal/mol): The classical energy function used in the Disulfide class
2. DSE (Disulfide Strain Energy) (kJ/mol): An alternative energy function with different components

Key features:
- Calculates individual energy components contributing to disulfide bond stability
- Provides detailed energy breakdowns for analysis and comparison
- Converts between energy units (kcal/mol and kJ/mol)
- Creates energy component plots showing contributions across dihedral angle ranges
- Generates surface plots to visualize energy landscapes across two chi angles

Usage examples:
    # Create an energy model with specific chi angles
    energy = DisulfideEnergy(-60, -60, -85, -60, -60)  # Left-Handed Spiral conformation

    # Get energy values
    print(f"Standard energy: {energy.standard_energy} kcal/mol")
    print(f"DSE energy: {energy.dse_energy} kJ/mol")

    # Display detailed energy component breakdown
    print(energy.summary())

    # Visualize energy components across a range of chi3 values
    energy.plot_energy_components_scan(angle_to_vary=3,
                                      filename="energy_scan.png",
                                      colorblind_friendly=True)

    # Create a surface plot varying chi2 and chi3
    DisulfideEnergy.create_surface_plot(chi_indices=(2, 3))

Author: Eric G. Suchanek, PhD
Last Revision: 2025-04-16 18:23:33
License: BSD
"""

# pylint: disable=C0301
# pylint: disable=C0103

import matplotlib.pyplot as plt
import numpy as np


class DisulfideEnergy:
    """
    Class to represent and calculate energy components of disulfide bonds.

    This class decomposes the energy functions used in the Disulfide class
    into their individual components for analysis and comparison.
    """

    def __init__(self, chi1=0.0, chi2=0.0, chi3=0.0, chi4=0.0, chi5=0.0):
        """
        Initialize with the five chi angles of a disulfide bond

        :param chi1: Chi1 dihedral angle in degrees
        :type chi1: float
        :param chi2: Chi2 dihedral angle in degrees
        :type chi2: float
        :param chi3: Chi3 dihedral angle in degrees
        :type chi3: float
        :param chi4: Chi4 dihedral angle in degrees
        :type chi4: float
        :param chi5: Chi5 dihedral angle in degrees
        :type chi5: float
        """
        self.set_dihedrals(chi1, chi2, chi3, chi4, chi5)

        # Conversion factors
        self.KCAL_TO_KJ = 4.184  # 1 kcal/mol = 4.184 kJ/mol
        self.KJ_TO_KCAL = 0.239006  # 1 kJ/mol = 0.239006 kcal/mol

    def set_dihedrals(self, chi1=None, chi2=None, chi3=None, chi4=None, chi5=None):
        """
        Set the dihedral angles for the disulfide bond

        :param chi1: Dihedral angle in degrees. If None, keeps current value
        :type chi1: float or None
        :param chi2: Dihedral angle in degrees. If None, keeps current value
        :type chi2: float or None
        :param chi3: Dihedral angle in degrees. If None, keeps current value
        :type chi3: float or None
        :param chi4: Dihedral angle in degrees. If None, keeps current value
        :type chi4: float or None
        :param chi5: Dihedral angle in degrees. If None, keeps current value
        :type chi5: float or None
        """
        if chi1 is not None:
            self._chi1 = float(chi1)
        if chi2 is not None:
            self._chi2 = float(chi2)
        if chi3 is not None:
            self._chi3 = float(chi3)
        if chi4 is not None:
            self._chi4 = float(chi4)
        if chi5 is not None:
            self._chi5 = float(chi5)

    def set_dihedrals_list(self, dihedrals):
        """
        Set the dihedral angles from a list

        :param dihedrals: List of 5 dihedral angles [chi1, chi2, chi3, chi4, chi5] in degrees
        :type dihedrals: list
        :raises ValueError: If the list does not contain exactly 5 dihedral angles
        """
        if len(dihedrals) != 5:
            raise ValueError("Expected a list of 5 dihedral angles")
        self.set_dihedrals(
            dihedrals[0], dihedrals[1], dihedrals[2], dihedrals[3], dihedrals[4]
        )

    @property
    def dihedrals(self):
        """
        :return: The dihedral angles as a list
        :rtype: list
        """
        return [self._chi1, self._chi2, self._chi3, self._chi4, self._chi5]

    @dihedrals.setter
    def dihedrals(self, angles):
        """
        :param angles: List of 5 dihedral angles
        :type angles: list
        """
        self.set_dihedrals_list(angles)

    def _torad(self, deg):
        """
        :param deg: Angle in degrees
        :type deg: float
        :return: Angle in radians
        :rtype: float
        """
        return np.deg2rad(deg)

    def calculate_standard_components(self):
        """
        Calculate the components of the standard energy function

        :return: Dictionary containing the components and their values in kcal/mol
        :rtype: dict
        """
        # Standard energy components (kcal/mol)
        std_components = {
            "chi1_chi5": 2.0
            * (
                np.cos(self._torad(3.0 * self._chi1))
                + np.cos(self._torad(3.0 * self._chi5))
            ),
            "chi2_chi4": np.cos(self._torad(3.0 * self._chi2))
            + np.cos(self._torad(3.0 * self._chi4)),
            "chi3": 3.5 * np.cos(self._torad(2.0 * self._chi3))
            + 0.6 * np.cos(self._torad(3.0 * self._chi3)),
            "constant": 10.1,
        }
        return std_components

    def calculate_dse_components(self):
        """
        Calculate the components of the DSE energy function

        :return: Dictionary containing the components and their values in kJ/mol
        :rtype: dict
        """
        # DSE components (kJ/mol)
        dse_components = {
            "chi1_chi5": 8.37 * (1 + np.cos(self._torad(3 * self._chi1)))
            + 8.37 * (1 + np.cos(self._torad(3 * self._chi5))),
            "chi2_chi4": 4.18 * (1 + np.cos(self._torad(3 * self._chi2)))
            + 4.18 * (1 + np.cos(self._torad(3 * self._chi4))),
            "chi3_2fold": 14.64 * (1 + np.cos(self._torad(2 * self._chi3))),
            "chi3_3fold": 2.51 * (1 + np.cos(self._torad(3 * self._chi3))),
            "constant": 0.0,
        }
        return dse_components

    def calculate_dse_components_kcal(self):
        """
        Calculate the components of the DSE energy function in kcal/mol

        :return: Dictionary containing the components and their values in kcal/mol
        :rtype: dict
        """
        dse_components = self.calculate_dse_components()
        return {k: v * self.KJ_TO_KCAL for k, v in dse_components.items()}

    @property
    def standard_energy(self):
        """
        :return: Total standard energy in kcal/mol
        :rtype: float
        """
        components = self.calculate_standard_components()
        return sum(components.values())

    @property
    def dse_energy(self):
        """
        :return: Total DSE energy in kJ/mol
        :rtype: float
        """
        components = self.calculate_dse_components()
        return sum(components.values())

    @property
    def dse_energy_kcal(self):
        """
        :return: Total DSE energy in kcal/mol
        :rtype: float
        """
        return self.dse_energy * self.KJ_TO_KCAL

    def get_all_components(self, units="mixed"):
        """
        Get all energy components with specified units

        :param units: Units for the energy values. One of:
                      'mixed' (default): Standard in kcal/mol, DSE in kJ/mol
                      'kcal': All values in kcal/mol
                      'kj': All values in kJ/mol
        :type units: str
        :return: Dictionary with two keys 'standard' and 'dse', each containing
                a dictionary of component values in the specified units
        :rtype: dict
        :raises ValueError: If units is not one of 'mixed', 'kcal', or 'kj'
        """
        std_components = self.calculate_standard_components()

        if units == "mixed":
            dse_components = self.calculate_dse_components()
        elif units == "kcal":
            dse_components = self.calculate_dse_components_kcal()
        elif units == "kj":
            dse_components = self.calculate_dse_components()
            std_components = {k: v * self.KCAL_TO_KJ for k, v in std_components.items()}
        else:
            raise ValueError("units must be one of 'mixed', 'kcal', or 'kj'")

        return {"standard": std_components, "dse": dse_components}

    def __str__(self):
        """
        :return: String representation showing energy values
        :rtype: str
        """
        std_energy = self.standard_energy
        dse_energy = self.dse_energy
        dse_energy_kcal = self.dse_energy_kcal

        return (
            f"DisulfideEnergy(chi={self.dihedrals})\n"
            f"  Standard energy: {std_energy:.2f} kcal/mol\n"
            f"  DSE energy: {dse_energy:.2f} kJ/mol ({dse_energy_kcal:.2f} kcal/mol)"
        )

    def summary(self):
        """
        :return: Detailed summary of energy components as a formatted string
        :rtype: str
        """
        std_components = self.calculate_standard_components()
        dse_components = self.calculate_dse_components()
        dse_components_kcal = self.calculate_dse_components_kcal()

        lines = [
            f"DisulfideEnergy Summary for chi=[{self._chi1:.1f}, {self._chi2:.1f}, {self._chi3:.1f}, {self._chi4:.1f}, {self._chi5:.1f}]"
        ]
        lines.append("\nStandard Energy Components (kcal/mol):")
        for name, value in std_components.items():
            lines.append(f"  {name:10s}: {value:8.3f}")
        lines.append(f"  {'Total':10s}: {self.standard_energy:8.3f}")

        lines.append("\nDSE Energy Components (kJ/mol):")
        for name, value in dse_components.items():
            lines.append(f"  {name:10s}: {value:8.3f}")
        lines.append(f"  {'Total':10s}: {self.dse_energy:8.3f}")

        lines.append("\nDSE Energy Components (kcal/mol):")
        for name, value in dse_components_kcal.items():
            lines.append(f"  {name:10s}: {value:8.3f}")
        lines.append(f"  {'Total':10s}: {self.dse_energy_kcal:8.3f}")

        return "\n".join(lines)

    def plot_energy_components_scan(
        self,
        angle_to_vary=3,
        angle_range=None,
        fixed_angles=None,
        filename=None,
        colorblind_friendly=True,
        show_plot=True,
    ):
        """
        Create a plot showing energy components across a range of values for one chi angle.

        :param angle_to_vary: The chi angle to vary (1-5)
        :type angle_to_vary: int
        :param angle_range: The range of angles to plot. Default is (-180, 180) with 90 points.
        :type angle_range: tuple or list or ndarray or None
        :param fixed_angles: List of 5 chi angles [chi1, chi2, chi3, chi4, chi5] to use as fixed values.
                            Any None value will be set to the current corresponding angle in self.dihedrals.
                            The angle specified by angle_to_vary will be overridden by the scan range.
        :type fixed_angles: list or None
        :param filename: If provided, save the plot to this filename
        :type filename: str or None
        :param colorblind_friendly: If True, use a colorblind-friendly palette
        :type colorblind_friendly: bool
        :param show_plot: If True (default), display the plot
        :type show_plot: bool
        :return: (figure, axes), the matplotlib figure and axes objects
        :rtype: tuple
        :raises ValueError: If angle_to_vary is not between 1 and 5
        """
        # Validate angle_to_vary
        if angle_to_vary < 1 or angle_to_vary > 5:
            raise ValueError("angle_to_vary must be between 1 and 5")

        # Set up angle range
        if angle_range is None:
            angle_range = np.linspace(-180, 180, 90)

        # Set up fixed angles
        if fixed_angles is None:
            fixed_angles = list(self.dihedrals)  # Use current dihedrals as default
        else:
            # Fill in any None values from current dihedrals
            for i in range(5):
                if fixed_angles[i] is None:
                    fixed_angles[i] = self.dihedrals[i]

        # Set up component dictionaries to store results
        std_components = {
            "chi1_chi5": [],
            "chi2_chi4": [],
            "chi3": [],
            "constant": [],
            "total": [],
        }

        dse_components = {
            "chi1_chi5": [],
            "chi2_chi4": [],
            "chi3_2fold": [],
            "chi3_3fold": [],
            "constant": [],
            "total": [],
        }

        # Calculate energy components for each angle in the range
        for angle in angle_range:
            # Create a copy of the fixed angles and update the one we're varying
            chi_values = fixed_angles.copy()
            chi_values[angle_to_vary - 1] = angle

            # Set the angles and calculate energy components
            self.set_dihedrals_list(chi_values)

            # Get standard components
            std = self.calculate_standard_components()
            for key, value in std.items():
                if key != "total":
                    std_components[key].append(value)
            std_components["total"].append(self.standard_energy)

            # Get DSE components (convert to kcal/mol for consistency)
            dse = self.calculate_dse_components_kcal()
            for key, value in dse.items():
                if key != "total":
                    dse_components[key].append(value)
            dse_components["total"].append(self.dse_energy_kcal)

        # Create colorblind-friendly palette if requested
        if colorblind_friendly:
            colors = {
                "chi1_chi5": "#648FFF",  # Blue
                "chi2_chi4": "#DC267F",  # Magenta
                "chi3": "#008000",  # Green
                "chi3_2fold": "#008000",  # Green
                "chi3_3fold": "#785EF0",  # Purple
                "constant": "#FFB000",  # Gold
                "total": "#000000",  # Black
            }

            # Line styles
            styles = {
                "chi1_chi5": "-",
                "chi2_chi4": "--",
                "chi3": "-.",
                "chi3_2fold": "-.",
                "chi3_3fold": ":",
                "constant": (0, (5, 1)),  # Dense dashed
                "total": "-",
            }
        else:
            # Default matplotlib colors
            colors = {
                "chi1_chi5": "blue",
                "chi2_chi4": "red",
                "chi3": "green",
                "chi3_2fold": "green",
                "chi3_3fold": "purple",
                "constant": "orange",
                "total": "black",
            }

            # Simple line styles
            styles = {
                "chi1_chi5": "-",
                "chi2_chi4": "--",
                "chi3": "-",
                "chi3_2fold": "-",
                "chi3_3fold": "--",
                "constant": ":",
                "total": "-",
            }

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot standard energy components
        for component, values in std_components.items():
            linewidth = 3 if component == "total" else 2
            ax1.plot(
                angle_range,
                values,
                label=component,
                color=colors.get(component, "gray"),
                linestyle=styles.get(component, "-"),
                linewidth=linewidth,
            )

        ax1.set_title(
            f"Standard Energy Components - chi{angle_to_vary} varied (kcal/mol)"
        )
        ax1.set_ylabel("Energy (kcal/mol)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot DSE energy components
        for component, values in dse_components.items():
            linewidth = 3 if component == "total" else 2
            ax2.plot(
                angle_range,
                values,
                label=component,
                color=colors.get(component, "gray"),
                linestyle=styles.get(component, "-"),
                linewidth=linewidth,
            )

        ax2.set_title(f"DSE Energy Components - chi{angle_to_vary} varied (kcal/mol)")
        ax2.set_xlabel(f"Chi{angle_to_vary} Angle (degrees)")
        ax2.set_ylabel("Energy (kcal/mol)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        # Show the plot if requested
        if show_plot:
            plt.show()

        return fig, (ax1, ax2)

    def plot_energy_components_all_angles(
        self,
        angle_range=None,
        fixed_angles=None,
        filename=None,
        colorblind_friendly=True,
        show_plot=True,
    ):
        """
        Create a single plot showing energy components across all 5 dihedral angles.

        This function calls plot_energy_components_scan for each dihedral angle and
        combines the results into a single figure with two subplots (standard and DSE energy).

        :param angle_range: The range of angles to plot. Default is (-180, 180) with 90 points.
        :type angle_range: tuple or list or ndarray or None
        :param fixed_angles: List of 5 chi angles [chi1, chi2, chi3, chi4, chi5] to use as fixed values.
                            Any None value will be set to the current corresponding angle in self.dihedrals.
        :type fixed_angles: list or None
        :param filename: If provided, save the plot to this filename
        :type filename: str or None
        :param colorblind_friendly: If True, use a colorblind-friendly palette
        :type colorblind_friendly: bool
        :param show_plot: If True (default), display the plot
        :type show_plot: bool
        :return: (figure, axes), the matplotlib figure and axes objects
        :rtype: tuple
        """
        # Set up angle range
        if angle_range is None:
            angle_range = np.linspace(-180, 180, 90)

        # Set up fixed angles
        if fixed_angles is None:
            fixed_angles = list(self.dihedrals)  # Use current dihedrals as default
        else:
            # Fill in any None values from current dihedrals
            for i in range(5):
                if fixed_angles[i] is None:
                    fixed_angles[i] = self.dihedrals[i]

        # Create a master figure with 5 rows and 2 columns
        # Each row will contain the plots for one chi angle
        master_fig = plt.figure(figsize=(12, 25))

        # Create a 5x2 grid for the subplots
        gs = master_fig.add_gridspec(5, 2)

        # Lists to store all axes for later reference
        all_axes = []

        # For each dihedral angle
        for i in range(1, 6):
            # Calculate the row for this chi angle (0-based)
            row = i - 1

            # Create a temporary copy of the current object
            temp_energy = DisulfideEnergy(*self.dihedrals)

            # Get the energy components for this angle
            temp_fig, (temp_ax1, temp_ax2) = temp_energy.plot_energy_components_scan(
                angle_to_vary=i,
                angle_range=angle_range,
                fixed_angles=fixed_angles,
                filename=None,  # Don't save individual plots
                colorblind_friendly=colorblind_friendly,
                show_plot=False,  # Don't show individual plots
            )

            # Create new axes in the master figure
            ax1 = master_fig.add_subplot(gs[row, 0])  # Standard energy plot
            ax2 = master_fig.add_subplot(gs[row, 1])  # DSE energy plot

            # Copy the contents from the temporary axes to the master figure axes, excluding the "constant" line
            for line in temp_ax1.get_lines():
                if line.get_label() != "constant":  # Skip the constant line
                    ax1.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        label=line.get_label(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                    )

            for line in temp_ax2.get_lines():
                if line.get_label() != "constant":  # Skip the constant line
                    ax2.plot(
                        line.get_xdata(),
                        line.get_ydata(),
                        label=line.get_label(),
                        color=line.get_color(),
                        linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(),
                        alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                    )

            # Copy titles and labels
            ax1.set_title(f"Standard Energy - Chi{i} varied (kcal/mol)")
            ax2.set_title(f"DSE Energy - Chi{i} varied (kcal/mol)")
            ax1.set_ylabel("Energy (kcal/mol)")
            ax2.set_ylabel("Energy (kcal/mol)")
            ax1.set_xlabel("Angle (degrees)")
            ax2.set_xlabel("Angle (degrees)")

            # Add legends to individual subplots with explicit location
            ax1.legend(loc="upper right", fontsize="small")
            ax2.legend(loc="upper right", fontsize="small")

            # Set y-axis limits for both plots to keep scales comparable
            ax1.set_ylim(top=10.0)
            ax2.set_ylim(top=10.0)

            # Add grid
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            # Store the axes
            all_axes.append((ax1, ax2))

            # Close the temporary figure to free memory
            plt.close(temp_fig)

        # Adjust layout
        plt.tight_layout()

        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        # Show the plot if requested
        if show_plot:
            plt.show()

        return master_fig, all_axes

    @staticmethod
    def create_surface_plot(chi_indices=(2, 3)):
        """
        Create a surface plot showing energy variation with two chi angles

        Parameters
        ----------
        chi_indices : tuple
            Indices of the two chi angles to vary (1-based, like chi1, chi2, etc.)
        """
        print(
            f"Creating surface plot for chi{chi_indices[0]} and chi{chi_indices[1]}..."
        )

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

        ax2.set_title(
            f"DSE Energy (kcal/mol) - chi{chi_indices[0]} vs chi{chi_indices[1]}"
        )
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


# class ends


if __name__ == "__main__":
    disulfide_energy = DisulfideEnergy(chi1=60, chi2=-60, chi3=-85, chi4=-60, chi5=-60)
    print(disulfide_energy)
    print(disulfide_energy.summary())
    disulfide_energy.plot_energy_components_scan(angle_to_vary=3)
    disulfide_energy.create_surface_plot(chi_indices=(2, 3))


# end of file
