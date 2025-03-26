# Analysis of Disulfide Bonds in Proteins of Known Structure
# Author: Eric G. Suchanek, PhD.
# Last revision: 2025-03-24 23:37:09 -egs-
# Cα Cβ Sγ

import logging
import os
import sys
from pathlib import Path

import numpy as np  # type: ignore
import pyvista as pv  # type: ignore

import proteusPy as pp
from proteusPy.ProteusGlobals import Torsion_DF_Cols

pv.set_jupyter_backend("trame")

HOME = Path.home()
PDB = Path(os.getenv("PDB", HOME / "pdb"))
PBAR_COLS = 78

pp.configure_master_logger("test_hexbin.log")
pp.set_logger_level_for_module("proteusPy", logging.INFO)

_logger = pp.create_logger(__name__)

pp.set_pyvista_theme("auto")


DPI = 300
WIDTH = 5.0
HEIGHT = 4.0
TORMIN = -179.0
TORMAX = 180.0
GRIDSIZE = 10


def plot_3d_hexbin_single(
    df: "pandas.DataFrame",
    column1: str,
    column2: str,
    width: int = 800,
    height: int = 800,
    gridsize: int = 30,
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


def plot_3d_hexbin_leftright(
    loader,
    width: int = 800,
    height: int = 600,
    gridsize: int = 30,
    tormin: float = -180.0,
    tormax: float = 180.0,
    scaling: str = "sqrt",
    column1: str = "chi2",
    column2: str = "chi4",
) -> None:
    """
    Create 3D hexbin plots for left and right-handed chi2-chi4 correlations with customizable z-scaling.

    :param loader: Loader object to retrieve torsion data
    :type loader: proteusPy.PDB_SS
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
    :default tormin: -180.0
    :param tormax: Maximum torsion angle
    :type tormax: float, optional
    :default tormax: 180.0
    :param scaling: Scaling method for z-values ('linear', 'sqrt', 'log', 'power')
    :type scaling: str, optional
    :default scaling: 'sqrt'
    :param column1: Name of the first column (x-axis)
    :type column1: str, optional
    :default column1: 'chi2'
    :param column2: Name of the second column (y-axis)
    :type column2: str, optional
    :default column2: 'chi4'
    """

    _SS_df = loader.getTorsions()

    _left = _SS_df["chi3"] <= 0.0
    _right = _SS_df["chi3"] > 0.0

    _SS_df_Left = _SS_df[_left]
    _SS_df_Right = _SS_df[_right]

    try:
        # Ensure width and height are integers
        width = int(width)
        height = int(height)

        # Extract data
        x_left = _SS_df_Left[column1]
        y_left = _SS_df_Left[column2]
        x_right = _SS_df_Right[column1]
        y_right = _SS_df_Right[column2]

        # Create 2D histogram bins for both datasets
        bins_left, xedges_left, yedges_left = np.histogram2d(
            x_left,
            y_left,
            bins=gridsize,
            range=[[tormin - 1, tormax + 1], [tormin - 1, tormax + 1]],
        )
        bins_right, xedges_right, yedges_right = np.histogram2d(
            x_right, y_right, bins=gridsize, range=[[tormin, tormax], [-180, 180]]
        )

        # Apply scaling to bin counts using match
        match scaling:
            case "linear":
                scaled_bins_left = bins_left.T
                scaled_bins_right = bins_right.T
                scale_label = "linear scale"
            case "sqrt":
                scaled_bins_left = np.sqrt(bins_left.T)
                scaled_bins_right = np.sqrt(bins_right.T)
                scale_label = "sqrt scale"
            case "log":
                scaled_bins_left = np.log1p(bins_left.T)
                scaled_bins_right = np.log1p(bins_right.T)
                scale_label = "log scale"
            case "power":
                power = 0.3
                scaled_bins_left = np.power(bins_left.T, power)
                scaled_bins_right = np.power(bins_right.T, power)
                scale_label = f"power scale ({power})"
            case _:
                raise ValueError(
                    f"Unsupported scaling method: {scaling}. Use 'linear', 'sqrt', 'log', or 'power'."
                )

        # Debug: Print min and max of scaled values
        print(
            f"Left plot - Min: {scaled_bins_left.min()}, Max: {scaled_bins_left.max()}"
        )
        print(
            f"Right plot - Min: {scaled_bins_right.min()}, Max: {scaled_bins_right.max()}"
        )

        # Create mesh grid for plotting
        x_grid, y_grid = np.meshgrid(xedges_left[:-1], yedges_left[:-1])
        x_grid_r, y_grid_r = np.meshgrid(xedges_right[:-1], yedges_right[:-1])

        # Create PyVista plotter with two subplots
        plotter = pv.Plotter(shape=(1, 2), window_size=[width * 2, height])

        # Left-handed plot (subplot 0)
        plotter.subplot(0, 0)
        grid_left = pv.StructuredGrid(x_grid, y_grid, scaled_bins_left)
        grid_left.point_data["Height"] = scaled_bins_left.ravel(order="F")
        plotter.add_mesh(
            grid_left,
            scalars="Height",
            cmap="nipy_spectral",
            show_edges=False,
            clim=[scaled_bins_left.min(), scaled_bins_left.max()],
            show_scalar_bar=False,  # Add this line to hide the scalar bar
        )
        plotter.add_title(
            f"{column1} - {column2} Correlation (Left-handed, {scale_label})",
            font_size=8,
        )
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

        # Right-handed plot (subplot 1)
        plotter.subplot(0, 1)
        grid_right = pv.StructuredGrid(x_grid_r, y_grid_r, scaled_bins_right)
        grid_right.point_data["Height"] = scaled_bins_right.ravel(order="F")
        plotter.add_mesh(
            grid_right,
            scalars="Height",
            cmap="nipy_spectral",
            show_edges=False,
            clim=[scaled_bins_right.min(), scaled_bins_right.max()],
            show_scalar_bar=False,  # Add this line to hide the scalar bar
        )
        plotter.add_title(
            f"{column1} - {column2} Correlation (Right-handed, {scale_label})",
            font_size=8,
        )
        plotter.show_grid()
        plotter.view_xy()
        # Add axes with custom labels
        plotter.add_axes(
            xlabel=column1,
            ylabel=column2,
            zlabel="Incidence",
            line_width=2,
            color="black",
            interactive=True,
        )

        plotter.enable_parallel_projection()
        plotter.add_scalar_bar(
            title="Incidence",
            vertical=True,
            position_x=0.9,  # Right side
            position_y=0.25,
            width=0.05,
            height=0.5,
            title_font_size=14,
            label_font_size=12,
        )

        # Final adjustments
        plotter.reset_camera()
        plotter.link_views()
        plotter.show(
            jupyter_backend="pythreejs" if "jupyter" in str(type(plotter)) else "trame"
        )

    except AttributeError as e:
        print(f"Error: DataFrame might be missing required columns (chi2, chi4): {e}")
    except ValueError as e:
        print(f"Error: Invalid parameter value: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(
            "Please ensure WIDTH and HEIGHT are integers and DataFrames contain valid data."
        )


# Example usage in Jupyter:
# pv.set_jupyter_backend('pythreejs')  # Add this at the start of your notebook
# plot_3d_hexbin(SS_df_Left, SS_df_Right, scaling='sqrt')

# Run the plot
# plot_3d_hexbin_single(
#    SS_df_Left, "chi2", "chi4", scaling="sqrt", width=1024, height=1024, gridsize=50
# )


def main():
    # default parameters will read from the package itself.
    PDB_SS = pp.Load_PDB_SS(verbose=True, subset=False, cutoff=-1.0, sg_cutoff=-1.0)
    _SSdf = PDB_SS.getTorsions()

    _near = _SSdf["ca_distance"] <= 8.0

    SS_df = _SSdf[_near]

    SS_df = SS_df[Torsion_DF_Cols].copy()

    _left = SS_df["chi3"] <= 0.0
    _right = SS_df["chi3"] > 0.0

    SS_df_Left = SS_df[_left]
    SS_df_Right = SS_df[_right]

    print(f"Left Handed: {SS_df_Left.shape[0]}, Right Handed: {SS_df_Right.shape[0]}")

    PDB_SS.plot_3d_hexbin_leftright(
        scaling="sqrt",
        width=1024,
        height=1024,
        gridsize=50,
        column1="chi2",
        column2="chi4",
    )


if __name__ == "__main__":
    main()
    sys.exit()


# eof
