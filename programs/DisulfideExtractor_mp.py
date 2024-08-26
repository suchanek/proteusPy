"""
`DisulfideExtractor.py`

# Purpose:
This program encapsulates the steps needed to extract disulfides from the PDB file repository,
build the DisulfideLoader object, and save it into the proteusPy module data directory. This program
utilizes multiprocessing to speed up the extraction process.

# Processes:
* Extract: Extract SS bonds from the PDB raw files, with a cutoff of `cutoff` A.
* Build: Load the data from the extraction and save it as a compressed .pkl file.
* Update: Copy the `.pkl` files to the repo.
* Subset: Only extract and process the first 1000 Disulfides found in the PDB directory.

Author: Eric G. Suchanek, PhD.
Last revision: 8/24/24 -egs-
"""

import argparse
import datetime
import glob
import logging
import multiprocessing
import os
import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path
from shutil import copy

from colorama import Fore, Style, init
from tqdm import tqdm

from proteusPy import (
    Disulfide,
    DisulfideList,
    Extract_Disulfides,
    Extract_Disulfides_From_List,
    load_disulfides_from_id,
    remove_duplicate_ss,
    set_logger_level_for_module,
)
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    PROBLEM_ID_FILE,
    SS_PICKLE_FILE,
    SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_DICT_PICKLE_FILE,
    SS_SUBSET_PICKLE_FILE,
)

_logger = logging.getLogger("DisulfideExtractor")
_logger.setLevel(logging.INFO)

set_logger_level_for_module("proteusPy", logging.ERROR)

PBAR_COLS = 100

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

REPO_DATA = HOME_DIR / "repos" / "proteusPy" / "data"
if not REPO_DATA.is_dir():
    print(f"Error: The directory {REPO_DATA} does not exist.")
    sys.exit(1)

DATA_DIR = PDB_BASE / "data"
if not DATA_DIR.is_dir():
    print(f"Error: The directory {DATA_DIR} does not exist.")
    sys.exit(1)

good_pdb_fpath = DATA_DIR / GOOD_PDB_FILE

ent_files = glob.glob(str(PDB_DIR / "*.ent"))
pdb_id_list = [Path(f).stem[3:7] for f in ent_files]

num_ent_files = len(ent_files)

__version__ = "2.0.2"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"""\nproteusPy Disulfide Bond Extractor v{__version__}\n 
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
        default=True,
    )
    parser.add_argument(
        "--build",
        "-b",
        action="store_true",
        help="Build Disulfide loader",
        default=False,
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
        default=True,
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
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="Number of threads to use for extraction",
        default=8,
    )

    return parser.parse_args()


def extract_disulfides_chunk(args):
    (start_idx, end_idx, sslist, pdbdir, dist_cutoff, verbose, quiet, pbar_index) = args

    from proteusPy import DisulfideList, load_disulfides_from_id

    result_list = []
    global overall_pbar

    if quiet:
        _logger.setLevel(logging.ERROR)

    entrylist = sslist[start_idx:end_idx]

    task_pbar = tqdm(
        total=len(entrylist),
        desc=f"{Fore.BLUE}  Task {pbar_index+1:2}{Style.RESET_ALL}".ljust(10),
        position=pbar_index + 1,
        leave=False,
        ncols=PBAR_COLS,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.YELLOW, Style.RESET_ALL),
        mininterval=1.0,
    )

    for entry in entrylist:
        _sslist = load_disulfides_from_id(
            entry,
            model_numb=0,
            verbose=verbose,
            quiet=quiet,
            pdb_dir=pdbdir,
            cutoff=dist_cutoff,
        )

        if len(_sslist) > 0:
            sslist = remove_duplicate_ss(_sslist)
            result_list.extend(sslist)

        task_pbar.update(1)
        # overall_pbar.update(1)

    task_pbar.close()
    return result_list


def do_extract(verbose, full, subset, cutoff, prune, nthreads=6):
    ent_files = glob.glob(str(PDB_DIR / "*.ent"))
    num_ent_files = len(ent_files)
    sslist = [Path(f).stem[3:7] for f in ent_files]

    if full:
        chunk_size = num_ent_files // nthreads
    else:
        chunk_size = 1000 // nthreads

    res_list = DisulfideList([], "PDB_ALL_SS")

    pool_args = [
        (
            i * chunk_size,
            (i + 1) * chunk_size if i != nthreads - 1 else num_ent_files,
            sslist,
            PDB_DIR,
            cutoff,
            verbose,
            True,
            i,
        )
        for i in range(nthreads)
    ]

    with multiprocessing.Pool(nthreads) as pool:
        results = pool.map(extract_disulfides_chunk, pool_args)

    for result in results:
        res_list.extend(result)

    # save the disulfides to a pickle file
    if full:
        if verbose:
            print(f"Saving SS list to: {DATA_DIR / SS_PICKLE_FILE}")

        with open(DATA_DIR / SS_PICKLE_FILE, "wb") as f:
            pickle.dump(res_list, f)

    if subset:
        if verbose:
            print(f"Saving SS list to: {DATA_DIR / SS_SUBSET_PICKLE_FILE}")

        with open(DATA_DIR / SS_SUBSET_PICKLE_FILE, "wb") as f:
            pickle.dump(res_list, f)

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
        PDB_SS = DisulfideLoader(
            datadir=DATA_DIR, subset=False, verbose=verbose, cutoff=cutoff
        )
        PDB_SS.save(savepath=DATA_DIR, subset=False, cutoff=cutoff)

    if subset:
        if verbose:
            print(
                f"--> Building the packed loader for the Disulfide subset with cutoff: {cutoff}..."
            )
        PDB_SS = DisulfideLoader(
            datadir=DATA_DIR, subset=True, verbose=verbose, cutoff=cutoff
        )
        PDB_SS.save(savepath=DATA_DIR, subset=True, cutoff=cutoff)

    return


def update_repo(datadir):
    copy(Path(datadir) / LOADER_FNAME, Path(REPO_DATA))
    copy(Path(datadir) / LOADER_SUBSET_FNAME, Path(REPO_DATA))
    copy(Path(datadir) / SS_PICKLE_FILE, Path(REPO_DATA))


def do_stuff(
    all=False,
    extract=False,
    build=False,
    full=False,
    update=False,
    subset=False,
    verbose=False,
    cutoff=-1.0,
    prune=False,
    threads=8,
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
    _threads = threads

    if all:
        _extract = _build = _update = _subset = _full = True

    if _extract == True:
        if verbose:
            print(f"Extracting with cutoff: {cutoff}")

        do_extract(
            verbose=_verbose,
            full=_full,
            subset=_subset,
            cutoff=cutoff,
            prune=prune,
            nthreads=_threads,
        )
        print("\n")

    if _build == True:
        if verbose:
            print(f"Building with cutoff: {cutoff}")
        do_build(_verbose, _full, _subset, cutoff)

    if _update == True:
        if verbose:
            print(f"Copying SS files from: {DATA_DIR} to {REPO_DATA}")

        update_repo(DATA_DIR)

    return


def main():
    start = time.time()
    args = parse_arguments()
    set_logger_level_for_module("proteusPy", logging.ERROR)

    print(
        f"\nproteusPy DisulfideExtractor v{__version__}\n"
        f"PDB model directory:       {PDB_DIR}\n"
        f"Data directory:            {DATA_DIR}\n"
        f"Module data directory:     {MODULE_DATA}\n"
        f"Repo data directory:       {REPO_DATA}\n"
        f"Number of .ent files:      {num_ent_files}\n"
        f"Using cutoff:              {args.cutoff}\n"
        f"Extract:                   {args.extract}\n"
        f"Build:                     {args.build}\n"
        f"Update:                    {args.update}\n"
        f"Full:                      {args.full}\n"
        f"Subset:                    {args.subset}\n"
        f"Verbose:                   {args.verbose}\n"
        f"Threads:                   {args.threads}\n"
        f"Starting at:               {datetime.datetime.now()}\n"
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
        threads=args.threads,
    )

    end = time.time()

    elapsed_time = timedelta(seconds=end - start)

    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Processing completed in {int(hours)}:{int(minutes)}:{int(seconds)}")


if __name__ == "__main__":
    main()

# End of file
