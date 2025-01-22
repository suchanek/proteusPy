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
Last revision: 8/21/24 -egs-
"""

import argparse
import datetime
import glob
import logging
import os
import pickle
import sys
import threading
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

__version__ = "2.0.0"


def Extract_Disulfides_From_List_threaded(
    numb,
    verbose,
    quiet,
    pdbdir,
    baddir,
    datadir,
    picklefile,
    problemfile,
    dist_cutoff,
    prune,
    sslist,
    start_idx,
    end_idx,
    num_threads,
    result_list,
    pbar_index,
    overall_pbar,
) -> None:
    """
    Read the PDB files contained in ``pdbdir`` and create the .pkl files needed for the
    proteusPy.DisulfideLoader.DisulfideLoader class.
    The ```Disulfide``` objects are contained in a ```DisulfideList``` object and
    ```Dict``` within these files. In addition, .csv files containing all of
    the torsions for the disulfides and problem IDs are written. The optional
    ```dist_cutoff``` allows for removal of Disufides whose Cα-Cα distance is >
    than the cutoff value. If it's -1.0 then the function keeps all Disulfides.

    :param numb:           Number of entries to process, defaults to all
    :param verbose:        More messages
    :param quiet:          Turn off DisulfideConstruction warnings
    :param pdbdir:         Path to PDB files
    :param datadir:        Path to resulting .pkl files
    :param picklefile:     Name of the disulfide .pkl file
    :param problemfile:    Name of the .csv file containing problem ids
    :param dist_cutoff:    Ca distance cutoff to reject a Disulfide.
    :param prune:          Move bad files to bad directory, defaults to True
    """

    import shutil

    from proteusPy import DisulfideList, load_disulfides_from_id
    from proteusPy.ProteusGlobals import Torsion_DF_Cols

    if quiet:
        _logger.setLevel(logging.ERROR)

    bad_dir = baddir

    entrylist = []
    problem_ids = []
    bad = bad_dist = tot = cnt = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_list = DisulfideList([], "PDB_SS")
    All_ss_dict2 = {}  # new dict of pointers to indices

    cwd = os.getcwd()

    os.chdir(pdbdir)

    entrylist = sslist[start_idx:end_idx]
    update_freq = len(entrylist) // 100
    overall_update_freq = len(sslist) // 1000

    if verbose:
        _logger.info(
            f"Extract_Disulfides(): PDB Ids: {entrylist}, len: {len(entrylist)}"
        )

    # define a tqdm progressbar using the fully loaded entrylist list.
    # If numb is passed then
    # only do the last numb entries.

    # loop over ss_filelist, create disulfides and initialize them
    # the logging_redirect_tqdm() context manager will redirect the logging output
    # to the tqdm progress bar.

    task_pbar = tqdm(
        total=len(entrylist),
        desc=f"{Fore.BLUE}  Task {pbar_index+1:2}{Style.RESET_ALL}".ljust(10),
        position=pbar_index + 1,
        leave=False,
        ncols=PBAR_COLS,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.YELLOW, Style.RESET_ALL),
        mininterval=1.0,
    )

    idx = 0
    for entry in entrylist:
        _sslist = DisulfideList([], entry)

        # returns an empty list if none are found.
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
            sslist2 = []  # list to hold indices for ss_dict2
            for ss in sslist:
                result_list.append(ss)
                cnt += 1
                tot += 1

        else:  ## _sslist is empty!
            bad += 1
            if verbose:
                _logger.warning(f"Extract_Disulfides(): No SS parsed for: {entry}!")
            continue  ## this entry has no SS bonds, so we break the loop and move on to the next entry

        idx += 1

        if idx % update_freq == 0:
            task_pbar.set_postfix({"ID": entry, "NoSS": bad, "Cnt": tot})
            task_pbar.update(update_freq)

        if idx % overall_update_freq == 0:
            overall_pbar.update(overall_update_freq)

    task_pbar.close()

    # return to original directory
    os.chdir(cwd)

    # restore the logger level
    if quiet:
        _logger.setLevel(logging.WARNING)

    return


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
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="Number of threads to use for extraction",
        default=4,
    )

    return parser.parse_args()


# total extraction uses numb=-1 and takes about 1.5 hours on
# a 2021 MacbookPro M1 Pro computer, ~50 minutes on a 2023 M3 Max MacbookPro.
# Using the parser with ssparser.py reduces time to approximately 19 minutes on the
# M3 Max.


def do_extract(verbose, full, subset, cutoff, prune, nthreads=6):
    from proteusPy import Extract_Disulfides, load_list_from_file

    ent_files = glob.glob(str(PDB_DIR / "*.ent"))
    num_ent_files = len(ent_files)
    sslist = [Path(f).stem[3:7] for f in ent_files]

    res_list = (
        DisulfideList([], "PDB_SS") if full else DisulfideList([], "PDB_SS_SUBSET")
    )

    threads = []
    if full:
        chunk_size = num_ent_files // nthreads
    else:
        chunk_size = 1000 // nthreads

    result_lists = [[] for _ in range(nthreads)]

    # Create the overall progress bar
    overall_pbar = tqdm(
        total=num_ent_files,
        desc=f"{Fore.GREEN}Overall Progress{Style.RESET_ALL}".ljust(20),
        position=0,
        leave=True,
        ncols=PBAR_COLS,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.GREEN, Style.RESET_ALL),
        mininterval=0.1,
    )

    for i in range(nthreads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != nthreads - 1 else num_ent_files
        pbar_index = i + 1  # so the task pbar is displayed in the correct position

        thread = threading.Thread(
            target=Extract_Disulfides_From_List_threaded,
            args=(
                -1,
                False,
                True,
                PDB_DIR,
                Path(PDB_DIR) / "bad",
                DATA_DIR,
                SS_PICKLE_FILE,
                PROBLEM_ID_FILE,
                cutoff,
                False,
                sslist,
                start_idx,
                end_idx,
                nthreads,
                result_lists[i],
                i,
                overall_pbar,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for result_list in result_lists:
        res_list.extend(result_list)

    overall_pbar.close()

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
    prune=False,
    threads=4,
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
