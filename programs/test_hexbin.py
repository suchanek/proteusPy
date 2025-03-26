# Analysis of Disulfide Bonds in Proteins of Known Structure
# Author: Eric G. Suchanek, PhD.
# Last revision: 2025-03-24 23:37:09 -egs-
# Cα Cβ Sγ

import logging
import sys

import numpy as np  # type: ignore
import pyvista as pv  # type: ignore

import proteusPy as pp
from proteusPy.ProteusGlobals import Torsion_DF_Cols

# pv.set_jupyter_backend("trame")

pp.configure_master_logger("test_hexbin.log")
pp.set_logger_level_for_module("proteusPy", logging.INFO)

_logger = pp.create_logger(__name__)

pp.set_pyvista_theme("auto")


def plot_3d_hexbin_single(
    df: "pandas.DataFrame",
    column1: str,
    column2: str,
    width: int = 800,
    height: int = 800,
    gridsize: int = 80,
    tormin: float = -180.0,
    tormax: float = 180.0,
    scaling: str = "sqrt",
) -> None:
    """
    Create a 3D hexbin plot for correlations between two columns from a
    single DataFrame with customizable z-scaling.

    :param df: Data containing the specified columns
    :type df: pandas.DataFrame
    :param column1: Name of the first column (x-axis)
    :type column1: str
    :param column2: Name of the second column (y-axis)
    :type column2: str
    :param width: Window width in pixels
    :type width: int, optional
    :default width: 800
    :param height: Window height in pixels
    :type height: int, optional
    :default height: 600
    :param gridsize: Number of bins for hexbin
    :type gridsize: int, optional
    :default gridsize: 30
    :param tormin: Minimum torsion angle
    :type tormin: float, optional
    :default tormin: -180
    :param tormax: Maximum torsion angle
    :type tormax: float, optional
    :default tormax: 180
    :param scaling: Scaling method for z-values ('linear', 'sqrt', 'log', 'power')
    :type scaling: str, optional
    :default scaling: 'sqrt'
    """
    try:
        # Ensure width and height are integers
        width = int(width)
        height = int(height)

        # Extract data from specified columns
        x = df[column1]
        y = df[column2]

        # Create 2D histogram bins
        bins, xedges, yedges = np.histogram2d(
            x, y, bins=gridsize, range=[[tormin, tormax], [tormin, tormax]]
        )

        # Apply scaling to bin counts using match
        match scaling:
            case "linear":
                scaled_bins = bins.T
                scale_label = "linear scale"
            case "sqrt":
                scaled_bins = np.sqrt(bins.T)
                scale_label = "sqrt scale"
            case "log":
                scaled_bins = np.log1p(bins.T)
                scale_label = "log scale"
            case "power":
                power = 0.3
                scaled_bins = np.power(bins.T, power)
                scale_label = f"power scale ({power})"
            case _:
                raise ValueError(
                    f"Unsupported scaling method: {scaling}. Use 'linear', 'sqrt', 'log', or 'power'."
                )

        # Debug: Print min and max of scaled values
        print(f"Plot - Min: {scaled_bins.min()}, Max: {scaled_bins.max()}")

        # Create mesh grid for plotting
        x_grid, y_grid = np.meshgrid(xedges[:-1], yedges[:-1])

        # Create PyVista plotter (single view)
        plotter = pv.Plotter(window_size=[width, height])

        # Create and plot the grid
        grid = pv.StructuredGrid(x_grid, y_grid, scaled_bins)
        grid.point_data["Height"] = scaled_bins.ravel(order="F")
        plotter.add_mesh(
            grid,
            scalars="Height",
            cmap="nipy_spectral",
            show_edges=False,
            clim=[scaled_bins.min(), scaled_bins.max()],
            show_scalar_bar=False,  # Add this line to hide the scalar bar
        )
        plotter.add_title(
            f"{column1} - {column2} Correlation ({scale_label})", font_size=8
        )

        # Add grid
        plotter.show_grid()
        # Add axes with custom labels
        plotter.add_axes(
            xlabel=column1,
            ylabel=column2,
            zlabel="Incidence",
            line_width=2,
            color="black",
            interactive=True,
        )
        plotter.view_xy()
        plotter.enable_parallel_projection()
        # Scalar bar on the right side with smaller text
        plotter.add_scalar_bar(
            title="Incidence",
            vertical=True,
            position_x=0.9,  # Right side
            position_y=0.25,
            width=0.05,
            height=0.5,
            title_font_size=14,
            label_font_size=8,
        )

        # Final adjustments
        plotter.reset_camera()
        # Handle native vs Jupyter rendering explicitly
        if "jupyter" in str(type(plotter)):
            backend = "pythreejs"
            _logger.info("Using Jupyter backend: %s", backend)
            plotter.show(jupyter_backend=backend)
        else:
            _logger.info("Using native VTK backend")
            plotter.show()  # Native rendering

    except AttributeError as e:
        print(
            f"Error: DataFrame might be missing required columns ({column1}, {column2}): {e}"
        )
    except ValueError as e:
        print(f"Error: Invalid parameter value: {e}")

    return


def main():
    # default parameters will read from the package itself.
    PDB = pp.Load_PDB_SS(verbose=True, subset=False, cutoff=-1.0, sg_cutoff=-1.0)

    PDB.plot_3d_hexbin_leftright(
        scaling="sqrt",
        width=1024,
        height=1024,
        gridsize=90,
        column1="chi2",
        column2="chi4",
    )

    pp.DisulfideVisualization.plot_3d_hexbin_df(
        df=PDB.TorsionDF,
        column1="chi2",
        column2="chi4",
        width=1024,
        height=1024,
        gridsize=80,
        scaling="sqrt",
    )


if __name__ == "__main__":
    main()
    sys.exit()


# eof
