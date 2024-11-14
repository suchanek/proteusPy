"""
bootstrap.py
Program downloads the proteusPy master Disulfide list and
builds the proteusPy database.
"""

import argparse

from proteusPy import Bootstrap_PDB_SS, DisulfideLoader, create_logger
from proteusPy.ProteusGlobals import DATA_DIR

_logger = create_logger(__name__)

__version__ = "0.1.0"


def parse_arguments():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description=f"""\nproteusPy Disulfide Bootstrapper v{__version__}\n
        This program downloads the master disulfide bond disulfide list and builds the DisulfideLoader objects."""
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
        default=True,
    )
    parser.add_argument(
        "--cutoff",
        "-c",
        type=float,
        help="Disulfide Distance Cutoff, (Angstrom)",
        default=8.0,
    )

    return parser.parse_args()


def main():
    """
    Main function.
    """

    args = parse_arguments()

    if args.verbose:
        print(f"Running proteusPy Disulfide Bootstrapper v{__version__}")
        print(f"Building master loader with cutoff: {args.cutoff}")

    Bootstrap_PDB_SS(verbose=args.verbose, subset=False, cutoff=args.cutoff, force=True)

    if args.verbose:
        print(f"Building subset loader with cutoff: {args.cutoff}")

    subset_loader = DisulfideLoader(
        datadir=DATA_DIR, subset=True, verbose=args.verbose, cutoff=args.cutoff
    )
    subset_loader.save(savepath=DATA_DIR, subset=True, cutoff=args.cutoff)

    if args.verbose:
        subset_loader.describe()


if __name__ == "__main__":
    main()

# End of file
