"""
`DisulfideExtractor.py`

# Purpose:
This program encapsulates the steps needed to extract disulfides from the PDB file repository,
build the DisulfideLoader object, and save it into the proteusPy module data directory.

# Processes:
* Extract: Extract SS bonds from the PDB raw files, with a cutoff of `cutoff` A.
* Build: Load the data from the extraction and save it as a compressed .pkl file.
* Update: Copy the `.pkl` files to the repo.
* Subset: Only extract and process the first 1000 Disulfides found in the PDB directory.

Author: Eric G. Suchanek, PhD.
Last revision: 7/23/24 -egs-
"""

import argparse
import datetime
import os
import sys
import time
from shutil import copy

from proteusPy import Extract_Disulfides, __version__
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_DICT_PICKLE_FILE,
    SS_SUBSET_PICKLE_FILE,
    SS_SUBSET_TORSIONS_FILE,
)

HOME_DIR = os.path.expanduser("~")
PDB_BASE = os.getenv("PDB")
PDB_DIR = MODULE_DATA = REPO_DATA = DATA_DIR = ""


if not os.path.isdir(PDB_BASE):
    print(f"Error: The directory {PDB_BASE} does not exist.")
    sys.exit(1)
else:
    print(f"Found PDB directory at: {PDB_BASE}  ")

PDB_DIR = os.path.join(PDB_BASE, "good/")
if not os.path.isdir(PDB_DIR):
    print(f"Error: The directory {PDB_DIR} does not exist.")
    sys.exit(1)

MODULE_DATA = os.path.join(HOME_DIR, "repos/proteusPy/proteusPy/data/")
if not os.path.isdir(MODULE_DATA):
    print(f"Error: The directory {MODULE_DATA} does not exist.")
    sys.exit(1)

REPO_DATA = os.path.join(HOME_DIR, "repos/proteusPy/data/")
if not os.path.isdir(REPO_DATA):
    print(f"Error: The directory {REPO_DATA} does not exist.")
    sys.exit(1)

DATA_DIR = os.path.join(PDB_BASE, "data/")
if not os.path.isdir(DATA_DIR):
    print(f"Error: The directory {DATA_DIR} does not exist.")
    sys.exit(1)

print(
    f"Using PDB models at: {PDB_DIR}\nData directory: {DATA_DIR}\nModule data directory: {MODULE_DATA}\nRepo data directory: {REPO_DATA}"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"""proteusPy v{__version__} Disulfide Bond Extractor.\n 
        This program extracts disulfide bonds from PDB files and builds a DisulfideLoader object.
        The program expects the environment variable PDB to be set to the base location of the PDB files.
        The PDB files are expected to be in the PDB/good directory. Relevant output files, (SS_*LOADER*.pkl) are stored in PDB/data."""
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process full and subset, create loaders, and update repo",
    )
    parser.add_argument("--extract", action="store_true", help="Extract data")
    parser.add_argument("--build", action="store_true", help="Build loader")
    parser.add_argument("--update", action="store_true", help="Update repo")
    parser.add_argument("--full", action="store_true", help="Process full SS database")
    parser.add_argument("--subset", action="store_true", help="Process SS subset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--cutoff", type=float, help="Disulfide Distance Cutoff, (Angstrom)"
    )

    parser.set_defaults(all=False)
    parser.set_defaults(update=False)
    parser.set_defaults(verbose=True)
    parser.set_defaults(extract=True)
    parser.set_defaults(subset=True)
    parser.set_defaults(build=True)
    parser.set_defaults(full=True)
    parser.set_defaults(cutoff=-1.0)
    return parser.parse_args()


def do_extract(verbose, full, subset, cutoff, homedir):
    if subset:
        if verbose:
            print("--> Extracting the SS subset...")

        Extract_Disulfides(
            numb=1000,
            pdbdir=PDB_DIR,
            datadir=DATA_DIR,
            dictfile=SS_SUBSET_DICT_PICKLE_FILE,
            picklefile=SS_SUBSET_PICKLE_FILE,
            torsionfile=SS_SUBSET_TORSIONS_FILE,
            problemfile=SS_PROBLEM_SUBSET_ID_FILE,
            verbose=False,
            quiet=True,
            dist_cutoff=cutoff,
        )

    # total extraction uses numb=-1 and takes about 1.5 hours on
    # a 2021 MacbookPro M1 Pro computer, ~50 minutes on a 2023 M3 Max MacbookPro

    if full:
        if verbose:
            print("--> Extracting the SS full dataset. This will take ~1.5 hours.")

        Extract_Disulfides(
            numb=-1,
            verbose=False,
            quiet=True,
            pdbdir=PDB_DIR,
            datadir=DATA_DIR,
            dist_cutoff=cutoff,
        )
    return


def do_build(verbose, full, subset, cutoff):
    """
    Load and save a ```proteusPy.DisulfideLoader``` object
    to a .pkl file.

    :param verbose: Verbosity, boolean
    :param full: Whether to load and save the full dataset, boolean
    :param subset: Whether to load and save the subset database, boolean
    """
    from proteusPy.DisulfideLoader import DisulfideLoader

    if full:
        if verbose:
            print(
                f"--> Building the packed loader for the full dataset with cutoff: {cutoff}..."
            )
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=False)
        PDB_SS.cutoff = cutoff
        PDB_SS.save(savepath=DATA_DIR, subset=False, cutoff=cutoff)

    if subset:
        if verbose:
            print(
                f"--> Building the packed loader for the Disulfide subset with cutoff: {cutoff}..."
            )
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=True)
        PDB_SS.cutoff = cutoff
        PDB_SS.save(savepath=DATA_DIR, subset=True, cutoff=cutoff)

    return


def do_stuff(
    all=False,
    extract=True,
    build=True,
    full=False,
    update=False,
    subset=True,
    verbose=True,
    cutoff=-1.0,
):
    """
    Main entrypoint for the proteusPy Disulfide database extraction and creation workflow.

    :param all: Extract the full database, defaults to False
    :param extract: Perform extaction from raw PDB files, defaults to False
    :param build: Build the loader(s), defaults to True
    :param full: Build the full loader, defaults to False
    :param update: Update the repo, defaults to True
    :param subset: Extract the subset, defaults to True
    :param verbose: Be noisy, defaults to True
    :param cutoff: Distance cutoff (A), defaults to -1.0 for full extract
    """
    _extract = extract
    _build = build
    _full = full
    _update = update
    _subset = subset
    _verbose = verbose

    if all:
        _extract = _build = _update = _subset = _full = True

    if _extract == True:
        print(f"Extracting with cutoff: {cutoff}")
        do_extract(_verbose, _full, _subset, cutoff, PDB_DIR)

    if _build == True:
        print(f"Building:")
        do_build(_verbose, _full, _subset, cutoff)

    if _update == True:
        print(f"Copying: {DATA_DIR} to {REPO_DATA}")
        # copy(f"{DATA_DIR}{LOADER_FNAME}", f"{MODULE_DATA}")
        # copy(f"{DATA_DIR}{LOADER_SUBSET_FNAME}", f"{MODULE_DATA}")
        copy(f"{DATA_DIR}{LOADER_FNAME}", f"{REPO_DATA}")
        copy(f"{DATA_DIR}{LOADER_SUBSET_FNAME}", f"{REPO_DATA}")
    return


def main():
    args = parse_arguments()

    print(f"DisulfideExtractor parsing {PDB_DIR} at: {datetime.datetime.now()}")
    start = time.time()

    do_stuff(
        all=args.all,
        extract=args.extract,
        build=args.build,
        update=args.update,
        full=args.full,
        subset=args.subset,
        verbose=args.verbose,
        cutoff=args.cutoff,
    )

    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# End of file
