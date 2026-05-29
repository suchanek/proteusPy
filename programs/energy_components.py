import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Define the energy component calculation functions
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


def plot_energy_components():
    # Create angle range for visualization
    angles = np.linspace(-180, 180, 360)

    # Define conversion factor from kJ/mol to kcal/mol
    KJ_TO_KCAL = 0.239006  # 1 kJ/mol = 0.239006 kcal/mol

    # Calculate energy components for each chi angle varied separately
    chi_data = {}

    for chi_to_vary in range(1, 6):
        chi_data[f"chi{chi_to_vary}"] = []
        for angle in angles:
            # Set up the chi angles (all at 0 except the one we're varying)
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
            std_components, dse_components = calculate_energy_components(
                chi1, chi2, chi3, chi4, chi5
            )
            chi_data[f"chi{chi_to_vary}"].append((std_components, dse_components))

    # Create figure with multiple subplots - one per chi angle
    fig, axs = plt.subplots(5, 2, figsize=(15, 20))

    # Colorblind-friendly palette
    # Using a combination of the IBM color-blind safe palette and Wong's palette
    # High contrast colors with distinct patterns
    colors = {
        "blue": "#648FFF",  # Blue
        "orange": "#DC267F",  # Pink/Magenta - highly distinguishable
        "yellow": "#FFB000",  # Gold/Yellow
        "green": "#008000",  # Strong green
        "black": "#000000",  # Black
        "red": "#FE6100",  # Orange/Vermillion - another highly distinguishable color
        "purple": "#785EF0",  # Purple
        "total": "#000000",  # Black for total energy
    }

    # Line styles to further differentiate
    line_styles = {
        "chi1_chi5": "-",  # Solid line
        "chi2_chi4": "--",  # Dashed line
        "chi3": "-.",  # Dash-dot line
        "chi3_2fold": "--",  # Dashed line
        "chi3_3fold": ":",  # Dotted line
        "constant": ":",  # Dotted line
        "total": "-",  # Solid thick line for total
    }

    # Line widths for emphasis
    line_widths = {
        "chi1_chi5": 2.0,
        "chi2_chi4": 2.0,
        "chi3": 2.0,
        "chi3_2fold": 2.0,
        "chi3_3fold": 2.0,
        "constant": 1.5,
        "total": 3.0,  # Thicker for total
    }

    # Colors for different components - colorblind friendly
    std_colors = {
        "chi1_chi5": colors["blue"],
        "chi2_chi4": colors["orange"],
        "chi3": colors["green"],
        "constant": colors["black"],
    }

    dse_colors = {
        "chi1_chi5": colors["blue"],
        "chi2_chi4": colors["orange"],
        "chi3_2fold": colors["green"],
        "chi3_3fold": colors["purple"],
    }

    # Add markers to further differentiate lines
    markers = {
        "chi1_chi5": "o",  # Circle
        "chi2_chi4": "s",  # Square
        "chi3": "^",  # Triangle
        "chi3_2fold": "d",  # Diamond
        "chi3_3fold": "x",  # X
        "constant": None,  # No marker
        "total": None,  # No marker for total
    }

    # Marker frequency (how often to show markers)
    marker_frequency = 30  # Show marker every 30 points (to avoid cluttering)

    # Plot for each chi angle
    for i in range(1, 6):
        chi_name = f"chi{i}"

        # Standard energy components (left column)
        ax_std = axs[i - 1, 0]
        for component_name in ["chi1_chi5", "chi2_chi4", "chi3", "constant"]:
            values = [
                chi_data[chi_name][j][0][component_name] for j in range(len(angles))
            ]

            # Plot with appropriate style, color, and markers
            if markers[component_name]:
                # Use markers sparingly to avoid cluttering
                markevery = slice(None, None, marker_frequency)
                ax_std.plot(
                    angles,
                    values,
                    label=component_name,
                    color=std_colors[component_name],
                    linestyle=line_styles[component_name],
                    linewidth=line_widths[component_name],
                    marker=markers[component_name],
                    markevery=markevery,
                    markersize=6,
                )
            else:
                ax_std.plot(
                    angles,
                    values,
                    label=component_name,
                    color=std_colors[component_name],
                    linestyle=line_styles[component_name],
                    linewidth=line_widths[component_name],
                )

        # Add total energy line
        total_values = [
            sum(chi_data[chi_name][j][0].values()) for j in range(len(angles))
        ]
        ax_std.plot(
            angles,
            total_values,
            label="Total",
            color=colors["total"],
            linestyle=line_styles["total"],
            linewidth=line_widths["total"],
        )

        ax_std.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax_std.set_title(f"Standard Energy Components - {chi_name} varied (kcal/mol)")
        ax_std.set_xlabel("Angle (degrees)")
        ax_std.set_ylabel("Energy (kcal/mol)")
        ax_std.legend()
        ax_std.grid(True, alpha=0.3)

        # DSE energy components (right column) - Convert kJ/mol to kcal/mol
        ax_dse = axs[i - 1, 1]
        for component_name in ["chi1_chi5", "chi2_chi4", "chi3_2fold", "chi3_3fold"]:
            # Convert from kJ/mol to kcal/mol
            values = [
                chi_data[chi_name][j][1][component_name] * KJ_TO_KCAL
                for j in range(len(angles))
            ]

            # Plot with appropriate style, color, and markers
            if markers[component_name]:
                # Use markers sparingly to avoid cluttering
                markevery = slice(None, None, marker_frequency)
                ax_dse.plot(
                    angles,
                    values,
                    label=component_name,
                    color=dse_colors[component_name],
                    linestyle=line_styles[component_name],
                    linewidth=line_widths[component_name],
                    marker=markers[component_name],
                    markevery=markevery,
                    markersize=6,
                )
            else:
                ax_dse.plot(
                    angles,
                    values,
                    label=component_name,
                    color=dse_colors[component_name],
                    linestyle=line_styles[component_name],
                    linewidth=line_widths[component_name],
                )

        # Add total energy line - Convert from kJ/mol to kcal/mol
        total_values = [
            sum(chi_data[chi_name][j][1].values()) * KJ_TO_KCAL
            for j in range(len(angles))
        ]
        ax_dse.plot(
            angles,
            total_values,
            label="Total",
            color=colors["total"],
            linestyle=line_styles["total"],
            linewidth=line_widths["total"],
        )

        ax_dse.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax_dse.set_title(
            f"DSE Components - {chi_name} varied (kcal/mol)"
        )  # Updated title to show kcal/mol
        ax_dse.set_xlabel("Angle (degrees)")
        ax_dse.set_ylabel("Energy (kcal/mol)")  # Updated y-axis label
        ax_dse.legend()
        ax_dse.grid(True, alpha=0.3)

    # Create a custom legend to explain the line styles and markers used
    custom_lines = [
        Line2D(
            [0],
            [0],
            color=colors["blue"],
            lw=2,
            linestyle="-",
            marker="o",
            markevery=5,
            label="chi1_chi5",
        ),
        Line2D(
            [0],
            [0],
            color=colors["orange"],
            lw=2,
            linestyle="--",
            marker="s",
            markevery=5,
            label="chi2_chi4",
        ),
        Line2D(
            [0],
            [0],
            color=colors["green"],
            lw=2,
            linestyle="-.",
            marker="^",
            markevery=5,
            label="chi3 (std) / chi3_2fold (DSE)",
        ),
        Line2D(
            [0],
            [0],
            color=colors["purple"],
            lw=2,
            linestyle=":",
            marker="x",
            markevery=5,
            label="chi3_3fold (DSE only)",
        ),
        Line2D(
            [0],
            [0],
            color=colors["black"],
            lw=1.5,
            linestyle=":",
            label="constant (std only)",
        ),
        Line2D([0], [0], color=colors["total"], lw=3, label="Total Energy"),
    ]

    # Add the custom legend to the bottom of the figure
    fig.legend(
        handles=custom_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=12,
        title="Component Guide - Colors, Line Styles, and Markers",
    )

    fig.suptitle("Disulfide Energy Components (Both Models in kcal/mol)", fontsize=16)

    plt.tight_layout()
    # Adjust layout to make room for the custom legend and suptitle
    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.savefig("energy_components.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_energy_components()
