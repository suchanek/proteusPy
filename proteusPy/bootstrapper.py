"""
Bootstrap proteusPy by building the DisulfideLoader

Author: Eric G. Suchanek, PhD
Last modification: 2025-02-13 19:19:27
"""

import argparse
import logging

import proteusPy as pp
from proteusPy.ProteusGlobals import CA_CUTOFF, DATA_DIR, SG_CUTOFF

VERSION = "1.0.2"

pp.set_logger_level_for_module("proteusPy", "INFO")
pp.configure_master_logger("bootstrap.log", log_level=logging.ERROR)


def main():
    """
    Main function to bootstrap proteusPy by building the DisulfideLoader.

    This function parses command-line arguments to set CA and SG cutoff values
    and verbosity. It then initializes and saves the total and subset
    DisulfideLoader with the specified parameters.

    Command-line arguments:
    -p, --percentile: Percentile cutoff value (default: -1.0)
    -v, --verbose: Enable verbose output

    The function performs the following steps:
    1. Parses command-line arguments.
    2. Prints verbose output if enabled.
    3. Initializes and saves the total DisulfideLoader.
    4. Initializes and saves the subset DisulfideLoader.
    """

    helpstring = """Bootstrap proteusPy by building the DisulfideLoader.
    This program downloads and builds the proteusPy DisulfideLoader object with
    specified Ca and Sg cutoffs. It then initializes and saves the total and subset
    DisulfideLoader with the specified parameters. Use cutoff values of -1 to build the database without filtering.
    """

    parser = argparse.ArgumentParser(
        description=helpstring,
    )
    parser.add_argument(
        "-p", "--percentile", type=float, default=-1.0, help="Percentile cutoff value"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print("proteusPy Bootstrapper v", VERSION)
    print("Percentile cutoff: ", args.percentile)
    print("Verbose: ", args.verbose)
    print("Data Directory: ", DATA_DIR)
    print("-> Downloading total Disulfide list from Google Drive")

    pdb_ss = pp.Bootstrap_PDB_SS(
        loadpath=DATA_DIR,
        verbose=args.verbose,
        subset=False,
        force=True,
        fake=False,
        percentile=args.percentile,
    )

    print("-> Saving Complete Loader")

    if pdb_ss is None:
        print("-> Failed to download/build DisulfideLoader!")
        return

    pdb_ss.save()

    print("-> Building Subset Loader")

    pdb_ss = pp.Bootstrap_PDB_SS(
        verbose=args.verbose,
        loadpath=DATA_DIR,
        percentile=args.percentile,
        subset=True,
        force=False,
        fake=False,
    )

    print("-> Saving subset Loader")

    pdb_ss.save(verbose=args.verbose)

    print("-> Complete and Subset Loaders built and saved!")
    return


if __name__ == "__main__":
    main()

# end of file
