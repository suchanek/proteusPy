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
Last revision: 7/10/24 -egs-
"""

import argparse
import datetime
import os
import time
from shutil import copy

from proteusPy import DisulfideLoader, Extract_Disulfides
from proteusPy.data import (
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

# location of cleaned PDB files, created with DisulfideDownloader.py
PDB_DIR = os.path.join(PDB_BASE, "good", "")

# this is specific to having a directory structure of ~/repos/proteusPy

MODULE_DATA = os.path.join(HOME_DIR, "repos/proteusPy/proteusPy/data/")
REPO_DATA = os.path.join(HOME_DIR, "repos/proteusPy/data/")

# location of the compressed Disulfide .pkl files
DATA_DIR = os.path.join(PDB_BASE, "data/")


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
    # my 2021 MacbookPro M1 Pro computer.

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
    Loads and saves a ```proteusPy.DisulfideLoader``` object
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
    all=True,
    extract=True,
    build=True,
    full=True,
    update=True,
    subset=True,
    verbose=True,
    cutoff=8.0,
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


parser = argparse.ArgumentParser()

parser.add_argument(
    "-a",
    "--all",
    help="do everything. Extract, build and save both datasets",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-c",
    "--cutoff",
    help="distance cutoff for disulfide distance pruning",
    type=float,
    required=False,
)
parser.add_argument(
    "-u",
    "--update",
    help="update the repo package",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="level of verbosity",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-e",
    "--extract",
    help="extract disulfides from the PDB structure files",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-f",
    "--full",
    help="extract all disulfides from the PDB structure files",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-b",
    "--build",
    help="rebuild the loader",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-s",
    "--subset",
    help="rebuild the subset only",
    action=argparse.BooleanOptionalAction,
)

parser.set_defaults(all=False)
parser.set_defaults(update=False)
parser.set_defaults(verbose=True)
parser.set_defaults(extract=True)
parser.set_defaults(subset=True)
parser.set_defaults(build=True)
parser.set_defaults(full=False)
parser.set_defaults(cutoff=8.0)

args = parser.parse_args()

all = args.all
extract = args.extract
build = args.build
update = args.update
full = args.full
subset = args.subset
verbose = args.verbose
cutoff = args.cutoff

print(f"DisulfideExtractor parsing {PDB_DIR}: {datetime.datetime.now()}")
start = time.time()

do_stuff(
    all=all,
    extract=extract,
    build=build,
    full=full,
    update=update,
    subset=subset,
    verbose=verbose,
    cutoff=cutoff,
)

end = time.time()
elapsed = end - start

print(
    f"DisulfideExtractor Complete!\nElapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)"
)

# End of file
