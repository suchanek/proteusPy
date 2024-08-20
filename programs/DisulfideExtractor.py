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
Last revision: 8/2/24 -egs-
"""

import argparse
import datetime
import glob
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from shutil import copy

from proteusPy import (
    Extract_Disulfides,
    Extract_Disulfides_From_List,
    set_logger_level_for_module,
)
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SS_DICT_PICKLE_FILE,
    SS_PICKLE_FILE,
    SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_DICT_PICKLE_FILE,
    SS_SUBSET_PICKLE_FILE,
    SS_SUBSET_TORSIONS_FILE,
    SS_TORSIONS_FILE,
)

_logger = logging.getLogger("DisulfideExtractor")
_logger.setLevel(logging.INFO)

set_logger_level_for_module("proteusPy", logging.ERROR)

PDB = os.getenv("PDB")
PDB_BASE = Path(PDB)

PDB_DIR = MODULE_DATA = REPO_DATA = DATA_DIR = ""
GOOD_PDB_FILE = "good_pdb.pkl"

PDB_BASE = Path(PDB)
HOME_DIR = Path.home()

if not PDB_BASE.is_dir():
    print(f"Error: The directory {PDB_BASE} does not exist.")
    sys.exit(1)

PDB_DIR = PDB_BASE / "good"
if not PDB_DIR.is_dir():
    print(f"Error: The directory {PDB_DIR} does not exist.")
    sys.exit(1)

MODULE_DATA = HOME_DIR / "repos" / "proteusPy" / "proteusPy" / "data"
if not MODULE_DATA.is_dir():
    print(f"Error: The directory {MODULE_DATA} does not exist.")
    sys.exit(1)

REPO_DATA = HOME_DIR / "repos/proteusPy/data"
if not REPO_DATA.is_dir():
    print(f"Error: The directory {REPO_DATA} does not exist.")
    sys.exit(1)

DATA_DIR = PDB_BASE / "data"
if not DATA_DIR.is_dir():
    print(f"Error: The directory {DATA_DIR} does not exist.")
    sys.exit(1)

good_pdb_fpath = DATA_DIR / GOOD_PDB_FILE
ent_files = glob.glob(str(PDB_DIR / "*.ent"))

num_ent_files = len(ent_files)
__version__ = "1.0.1"


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
        default=False,
    )
    parser.add_argument(
        "--extract",
        "-e",
        action="store_true",
        help="Extract Disulfide data",
        default=False,
    )
    parser.add_argument(
        "--build",
        "-b",
        action="store_true",
        help="Build Disulfide loader",
        default=True,
    )
    parser.add_argument(
        "--update",
        "-u",
        action="store_true",
        help="Update repository data directory",
        default=False,
    )
    parser.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Process full SS database",
        default=False,
    )
    parser.add_argument(
        "--subset", "-s", action="store_true", help="Process SS subset", default=False
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
        default=False,
    )
    parser.add_argument(
        "--cutoff",
        "-c",
        type=float,
        help="Disulfide Distance Cutoff, (Angstrom)",
        default=-1.0,
    )

    return parser.parse_args()


def do_extract(verbose, full, subset, cutoff, prune):
    from proteusPy import Extract_Disulfides, load_list_from_file

    # sslist = load_list_from_file(good_pdb_fpath)
    # print(f"len(sslist): {len(sslist)}")

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
            verbose=verbose,
            quiet=True,
            dist_cutoff=cutoff,
            prune=prune,
        )

    # total extraction uses numb=-1 and takes about 1.5 hours on
    # a 2021 MacbookPro M1 Pro computer, ~50 minutes on a 2023 M3 Max MacbookPro.
    # Using the parser with ssparser.py reduces time to approximately 19 minutes on the
    # M3 Max.

    if full:
        if verbose:
            print("--> Extracting the SS full dataset. This will take a while...")

        Extract_Disulfides(
            numb=-1,
            verbose=verbose,
            quiet=True,
            pdbdir=PDB_DIR,
            datadir=DATA_DIR,
            dist_cutoff=cutoff,
            prune=prune,
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
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=False, verbose=verbose)
        PDB_SS.cutoff = cutoff
        PDB_SS.save(savepath=DATA_DIR, subset=False, cutoff=cutoff)

    if subset:
        if verbose:
            print(
                f"--> Building the packed loader for the Disulfide subset with cutoff: {cutoff}..."
            )
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=True, verbose=verbose)
        PDB_SS.cutoff = cutoff
        PDB_SS.save(savepath=DATA_DIR, subset=True, cutoff=cutoff)

    return


def update_repo(datadir):
    copy(Path(datadir) / LOADER_FNAME, Path(REPO_DATA))
    copy(Path(datadir) / LOADER_SUBSET_FNAME, Path(REPO_DATA))
    copy(Path(datadir) / SS_DICT_PICKLE_FILE, Path(REPO_DATA))
    copy(Path(datadir) / SS_PICKLE_FILE, Path(REPO_DATA))
    copy(Path(datadir) / SS_SUBSET_DICT_PICKLE_FILE, Path(REPO_DATA))
    copy(Path(datadir) / SS_SUBSET_PICKLE_FILE, Path(REPO_DATA))
    return


def do_stuff(
    all=False,
    extract=False,
    build=False,
    full=False,
    update=False,
    subset=False,
    verbose=False,
    cutoff=-1.0,
    prune=True,
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
        if verbose:
            print(f"Extracting with cutoff: {cutoff}")
        do_extract(_verbose, _full, _subset, cutoff, prune)

    if _build == True:
        if verbose:
            print(f"Building:")
        do_build(_verbose, _full, _subset, cutoff)

    if _update == True:
        if verbose:
            print(f"Copying: {DATA_DIR} to {REPO_DATA}")

        update_repo(DATA_DIR)

    return


def main():
    start = time.time()
    args = parse_arguments()
    set_logger_level_for_module("proteusPy", logging.ERROR)

    print(
        f""
        f"proteusPy DisulfideExtractor v{__version__}\n"
        f"PDB model directory:       {PDB_DIR}\n"
        f"Data directory:            {DATA_DIR}\n"
        f"Module data directory:     {MODULE_DATA}\n"
        f"Repo data directory:       {REPO_DATA}\n"
        f"Number of .ent files:      {num_ent_files}\n"
        f"Using cutoff:              {args.cutoff}\n"
        f"Starting at:               {datetime.datetime.now()}\n\n"
    )

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
