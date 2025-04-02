"""
hexbin_plotter.py
Purpose: Create hexbin plots with customizable parameters.
Usage: python hexbin_plotter.py [arguments]
Author: Eric G. Suchanek, PhD.
Last revision: 2025-03-26
"""

# pylint: disable=c0103

import argparse
import logging
import sys

import proteusPy as pp


def setup_logging(log_file: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    :param log_file: Output log file name
    :param log_level: Logging level (default: "INFO")
    :return: Configured logger instance
    """
    pp.configure_master_logger(log_file)
    pp.set_logger_level_for_module("proteusPy", getattr(logging, log_level.upper()))
    return pp.create_logger(__name__)


def create_hexbin_plot(
    plot_type: str = "df",
    column1: str = "chi2",
    column2: str = "chi4",
    width: int = 1024,
    height: int = 1024,
    gridsize: int = 100,
    scaling: str = "sqrt",
    percentile: float = None,
    verbose: bool = False,
    title: str = None,
) -> None:
    """
    Create a hexbin plot with specified parameters.

    :param plot_type: Type of hexbin plot ('leftright' or 'df')
    :type plot_type: str
    :param column1: First column name for plotting
    :type column1: str
    :param column2: Second column name for plotting
    :type column2: str
    :param width: Plot width in pixels (default: 1024)
    :type width: int
    :param height: Plot height in pixels (default: 1024)
    :type height: int
    :param gridsize: Number of hexagons in grid (default: 100)
    :type gridsize: int
    :param scaling: Scaling method ('sqrt', 'linear', etc.) (default: 'sqrt')
    :type scaling: str
    :param percentile: Cutoff value for data loading (default: None)
    :type percentile: float
    :param verbose: Enable verbose output (default: False)
    :type verbose: bool
    """
    # Load PDB data
    if percentile > 0:
        PDB = pp.DisulfideLoader(verbose=verbose, percentile=percentile)
    else:
        PDB = pp.Load_PDB_SS(verbose=verbose, subset=False)

    # Create plot based on type
    if plot_type.lower() == "leftright":
        PDB.plot_3d_hexbin_leftright(
            scaling=scaling,
            width=width,
            height=height,
            gridsize=gridsize,
            column1=column1,
            column2=column2,
            title=title,
        )
    elif plot_type.lower() == "df":
        pp.DisulfideVisualization.plot_3d_hexbin_df(
            df=PDB.TorsionDF,
            column1=column1,
            column2=column2,
            width=width,
            height=height,
            gridsize=gridsize,
            scaling=scaling,
            title=title,
        )
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}. Use 'leftright' or 'df'")


def main():
    """Main function to parse arguments and create hexbin plot."""
    parser = argparse.ArgumentParser(description="Create hexbin plots with proteusPy")
    parser.add_argument(
        "plot_type", choices=["leftright", "df"], help="Type of hexbin plot to create"
    )
    parser.add_argument(
        "column1",
        choices=["chi1", "chi2", "chi3", "chi4", "chi5", "energy", "rho"],
        help="First column name for plotting",
    )
    parser.add_argument(
        "column2",
        choices=["chi1", "chi2", "chi3", "chi4", "chi5", "energy", "rho"],
        help="Second column name for plotting",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Plot width in pixels (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Plot height in pixels (default: 1024)"
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=80,
        help="Number of hexagons in grid (default: 100)",
    )
    parser.add_argument(
        "--scaling",
        default="sqrt",
        help="Scaling method (default: sqrt)",
        choices=["sqrt", "linear", "log", "power"],
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=-1.0,
        help="Percentile cutoff to build the dataset (default: -1.0)",
    )
    parser.add_argument(
        "--log-file",
        default="hexbin_plot.log",
        help="Output log file name (default: hexbin_plot.log)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output (default: False)"
    )

    args = parser.parse_args()

    # Setup environment and logging
    pp.set_pyvista_theme("auto")
    logger = setup_logging(args.log_file, args.log_level)

    try:
        create_hexbin_plot(
            plot_type=args.plot_type,
            column1=args.column1,
            column2=args.column2,
            width=args.width,
            height=args.height,
            gridsize=args.gridsize,
            scaling=args.scaling,
            percentile=args.percentile,
            verbose=args.verbose,
        )
        logger.info("Plot created successfully")
    except ValueError as e:
        logger.error("Error creating plot: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
    sys.exit(0)
