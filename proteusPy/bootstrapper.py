# Bootstrap proteusPy by building the DisulfideLoader
#
# Author: Eric G. Suchanek, PhD
# Last modification: 2025-01-26 17:35:43
#

import argparse
import logging

from proteusPy.DisulfideLoader import Bootstrap_PDB_SS
from proteusPy.logger_config import configure_master_logger, set_logger_level_for_module
from proteusPy.ProteusGlobals import CA_CUTOFF, SG_CUTOFF

version = "1.0.0"
set_logger_level_for_module("proteusPy", "INFO")
configure_master_logger("bootstrap.log", log_level=logging.ERROR)


def main():
    """
    Main function to bootstrap proteusPy by building the DisulfideLoader.

    This function parses command-line arguments to set CA and SG cutoff values
    and verbosity. It then initializes and saves the total and subset
    DisulfideLoader with the specified parameters.

    Command-line arguments:
    -c, --ca_cutoff: CA cutoff value (default: -1)
    -s, --sg_cutoff: SG cutoff value (default: -1)
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
        "-c", "--ca_cutoff", type=float, default=CA_CUTOFF, help="CA cutoff value"
    )
    parser.add_argument(
        "-s", "--sg_cutoff", type=float, default=SG_CUTOFF, help="SG cutoff value"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print("proteusPy Bootstrapper v", version)
    print("CA cutoff: ", args.ca_cutoff)
    print("SG cutoff: ", args.sg_cutoff)
    print("Verbose: ", args.verbose)
    print("Downloading total Disulfide list from Drive")

    pdb_ss = Bootstrap_PDB_SS(
        cutoff=args.ca_cutoff,
        sg_cutoff=args.sg_cutoff,
        verbose=args.verbose,
        subset=False,
        force=True,
    )

    print("Saving Complete Loader")

    pdb_ss.save(cutoff=args.ca_cutoff, sg_cutoff=args.sg_cutoff, subset=False)

    print("Building Subset Loader")

    pdb_ss = Bootstrap_PDB_SS(
        verbose=args.verbose,
        cutoff=args.ca_cutoff,
        sg_cutoff=args.sg_cutoff,
        subset=True,
        force=False,
    )

    print("Saving subset Loader")

    pdb_ss.save(cutoff=args.ca_cutoff, sg_cutoff=args.sg_cutoff, subset=True)

    print("Complete and Subset Loaders built and saved!")
    return


if __name__ == "__main__":
    main()

# end of file
