"""
test_hexbin.py
Purpose: Test hexbin plots for the proteusPy package.
Usage: python test_hexbin.py
Author: Eric G. Suchanek, PhD.
Last revision: 2025-03-26 18:12:46 -egs-
"""

# pylint: disable=c0103

import logging
import sys

import proteusPy as pp

# pv.set_jupyter_backend("trame")

# set to automatic theming
pp.set_pyvista_theme("auto")

# create a logger for the program
_logger = pp.create_logger(__name__)

# set up logging for the module
pp.configure_master_logger("test_hexbin.log")
pp.set_logger_level_for_module("proteusPy", logging.INFO)


def main():
    """Main function for testing hexbin plots."""

    # default parameters will read from the package itself.
    PDB = pp.Load_PDB_SS(verbose=True, subset=False, cutoff=-1.0, sg_cutoff=-1.0)

    PDB.plot_3d_hexbin_leftright(
        scaling="sqrt",
        width=1024,
        height=1024,
        gridsize=80,
        column1="chi1",
        column2="chi5",
    )

    pp.DisulfideVisualization.plot_3d_hexbin_df(
        df=PDB.TorsionDF,
        column1="chi1",
        column2="chi5",
        width=1024,
        height=1024,
        gridsize=80,
        scaling="sqrt",
    )


if __name__ == "__main__":
    main()
    sys.exit()


# eof
