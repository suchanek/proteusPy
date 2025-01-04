# pylint: disable=C0301 # line too long
# pylint: disable=C0413 # wrong import order
# pylint: disable=C0103 # invalid variable name

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
Last revision: 2025-01-01 09:17:35 -egs-
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

# Initialize colorama
init()

from tqdm import tqdm

from proteusPy import (
    DisulfideList,
    DisulfideLoader,
    configure_master_logger,
    create_logger,
    load_disulfides_from_id,
    remove_duplicate_ss,
    set_logger_level,
    toggle_stream_handler,
)
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SS_PICKLE_FILE,
    SS_SUBSET_PICKLE_FILE,
)

# Create a root logger. This will open ~/logs/DisulfideExtractor.log
# and write all log messages to it. There are quite a few messages
# generated while parsing. This provides a record of the process.

configure_master_logger("DisulfideExtractor.log")
set_logger_level("proteusPy.ssparser", "ERROR")
set_logger_level("proteusPy.DisulfideList", "INFO")
# set_logger_level("proteusPy.DisulfideLoader", "INFO")

# Disable the stream handlers for the following namespaces.
# This will suppress the output to the console.
toggle_stream_handler("proteusPy.ssparser", False)
toggle_stream_handler("proteusPy.DisulfideList", False)
toggle_stream_handler("proteusPy.DisulfideClass_Constructor", False)

# Create a logger for this program.
_logger = create_logger("DisulfideExtractor")
_logger.setLevel("INFO")


PBAR_COLS = 79

HOME_DIR = Path.home()

PDB = os.getenv("PDB")
PDB_BASE = Path(PDB)

PDB_DIR = MODULE_DATA = REPO_DATA = DATA_DIR = ""
GOOD_PDB_FILE = "good_pdb.pkl"

MINIFORGE_DIR = HOME_DIR / Path("miniforge3/envs")
MAMBAFORGE_DIR = HOME_DIR / Path("mambaforge/envs")

VENV_DIR = Path("lib/python3.12/site-packages/proteusPy/data")

PDB_BASE = Path(PDB)

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
num_ent_files = len(ent_files)

pdb_id_list = [Path(f).stem[3:7] for f in ent_files]


__version__ = "2.3.0"


def parse_arguments():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description=f"""\nproteusPy Disulfide Bond Extractor v{__version__}\n
        This program extracts disulfide bonds from PDB files and builds a DisulfideLoader object.
        The program expects the environment variable PDB to be set to the base location of the PDB files.
        The PDB files are expected to be in the PDB/good directory. Relevant output files, 
        (SS_*LOADER*.pkl) are stored in PDB/data."""
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
        default=True,
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
    parser.add_argument(
        "--forge",
        "-r",
        type=str,
        help="miniforge3 or mambaforge",
        default="miniforge3",
    )
    parser.add_argument(
        "--env",
        "-n",
        type=str,
        help="ppydev or proteusPy",
        default="ppydev",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=float,
        help="Sg cutoff distance",
        default=-1.0,
    )

    return parser.parse_args()


def extract_disulfides_chunk(args):
    "This is a single thread, processing one chunk of the multiprocessing extraction process."

    (
        start_idx,
        end_idx,
        sslist,
        pdbdir,
        dist_cutoff,
        verbose,
        quiet,
        pbar_index,
        sg_cutoff,
    ) = args

    result_list = []

    if quiet:
        _logger.setLevel(logging.WARNING)

    entrylist = sslist[start_idx:end_idx]

    task_pbar = tqdm(
        total=len(entrylist),
        desc=f"{Fore.BLUE}  Process {pbar_index+1:2}{Style.RESET_ALL}".ljust(10),
        position=pbar_index + 1,
        leave=False,
        ncols=PBAR_COLS,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.YELLOW, Style.RESET_ALL),
        mininterval=1.0,
    )

    for entry in entrylist:
        _sslist = load_disulfides_from_id(
            entry,
            verbose=verbose,
            quiet=quiet,
            pdb_dir=pdbdir,
            cutoff=dist_cutoff,
            sg_cutoff=sg_cutoff,
        )

        if len(_sslist) > 0:
            sslist = remove_duplicate_ss(_sslist)
            result_list.extend(sslist)

        task_pbar.update(1)

    task_pbar.close()
    return result_list


def do_extract(verbose, full, cutoff=-1.0, sg_cutoff=-1.0, nthreads=8):
    """
    Extracts the disulfides from the PDB files using multiprocessing.

    This function processes PDB files to extract disulfide bonds. It uses multiprocessing
    to parallelize the extraction process across multiple threads. The results are then
    combined and saved to a pickle file.

    :param verbose: If True, enables verbose output.
    :type verbose: bool
    :param full: If True, processes all PDB files; otherwise, processes a subset.
    :type full: bool
    :param cutoff: The cutoff value for filtering disulfides.
    :type cutoff: float
    :param prune: Not used in this function.
    :type prune: bool
    :param nthreads: The number of threads to use for multiprocessing. Default is 6.
    :type nthreads: int
    :return: None
    """

    sslist = [Path(f).stem[3:7] for f in ent_files]

    if full:
        _num_ent_files = num_ent_files
    else:
        _num_ent_files = 1000

    chunk_size = _num_ent_files // nthreads
    res_list = DisulfideList([], "PDB_ALL_SS")

    pool_args = [
        (
            i * chunk_size,
            (i + 1) * chunk_size if i != nthreads - 1 else _num_ent_files,
            sslist,
            PDB_DIR,
            cutoff,
            verbose,
            True,
            i,
            sg_cutoff,
        )
        for i in range(nthreads)
    ]

    with multiprocessing.Pool(nthreads) as pool:
        results = pool.map(extract_disulfides_chunk, pool_args)

    if verbose:
        print("\nProcessing results...")

    for result in results:
        res_list.extend(result)

    # save the disulfides to a pickle file
    if full:
        if verbose:
            print(f"Saving SS list to: {DATA_DIR / SS_PICKLE_FILE}")

        with open(str(DATA_DIR / SS_PICKLE_FILE), "wb+") as f:
            pickle.dump(res_list, f)

    else:
        if verbose:
            print(f"Saving SS subset list to: {DATA_DIR / SS_SUBSET_PICKLE_FILE}")

        with open(str(DATA_DIR / SS_SUBSET_PICKLE_FILE), "wb+") as f:
            pickle.dump(res_list, f)


def do_build(verbose, full, subset, cutoff, sg_cutoff):
    """
    Load and save a ```proteusPy.DisulfideLoader``` object
    to a .pkl file.

    :param verbose: Verbosity, boolean
    :param full: Whether to load and save the full dataset, boolean
    :param subset: Whether to load and save the subset database, boolean
    """

    if full:
        if verbose:
            print(
                f"Building the compressed loader for the full dataset with cutoffs: Cα: {cutoff}Å, Sγ: {sg_cutoff}Å"
            )
        PDB_SS = DisulfideLoader(
            datadir=DATA_DIR,
            subset=False,
            verbose=verbose,
            cutoff=cutoff,
            sg_cutoff=sg_cutoff,
        )
        PDB_SS.save(
            savepath=DATA_DIR, subset=subset, cutoff=cutoff, sg_cutoff=sg_cutoff
        )

    elif subset:
        if verbose:
            print(
                f"Building the packed loader for the Disulfide subset with cutoffs: {cutoff}, {sg_cutoff}..."
            )
        PDB_SS = DisulfideLoader(
            datadir=DATA_DIR,
            subset=True,
            verbose=verbose,
            cutoff=cutoff,
            sg_cutoff=sg_cutoff,
        )
        PDB_SS.save(
            savepath=DATA_DIR, subset=subset, cutoff=cutoff, sg_cutoff=sg_cutoff
        )
    else:
        print("Error: No valid build option selected.")
        sys.exit(1)

    PDB_SS.describe()


def update_repo(datadir, destdir):
    """
    Updates the repository with the latest SS files.
    """

    copy(Path(datadir) / LOADER_FNAME, Path(destdir))
    copy(Path(datadir) / LOADER_SUBSET_FNAME, Path(destdir))
    copy(Path(datadir) / SS_PICKLE_FILE, Path(destdir))
    copy(Path(datadir) / SS_SUBSET_PICKLE_FILE, Path(destdir))


def do_stuff(
    extract=False,
    build=False,
    full=False,
    update=False,
    subset=False,
    verbose=False,
    cutoff=-1.0,
    threads=8,
    forge="miniforge3",
    env="ppydev",
    sg_cutoff=-1.0,
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
    _forge = forge
    _env = env
    _cutoff = cutoff

    _master = bool(_cutoff < 0.0)

    if _extract is True:
        do_extract(
            verbose=_verbose,
            full=_full,
            cutoff=cutoff,
            sg_cutoff=sg_cutoff,
            nthreads=_threads,
        )
        print("\n")

    if _build is True:
        if verbose:
            if _master:
                print("Building master loader since cutoff is negative.")
            else:
                print(f"Building with Ca cutoff: {cutoff}, Sg cutoff: {sg_cutoff}")
        do_build(_verbose, _full, _subset, cutoff, sg_cutoff)

    if _update is True:

        if _forge == "miniforge3":
            venv_dir = MINIFORGE_DIR / _env / VENV_DIR
        else:
            venv_dir = MAMBAFORGE_DIR / _env / VENV_DIR

        if verbose:
            print(f"Copying SS files from: {DATA_DIR} to {venv_dir}")

        update_repo(DATA_DIR, venv_dir)

    return


def clear_screen():
    """
    Clears the terminal screen.
    """
    if os.name == "nt":  # For Windows
        os.system("cls")
    else:  # For macOS and Linux
        os.system("clear")


def main():
    "Main entrypoint for the DisulfideExtractor program."
    clear_screen()
    start = time.time()
    args = parse_arguments()

    # set_logging_level_for_all_handlers(logging.ERROR)
    # toggle_stream_handler("proteusPy.ssparser", False)

    # set_logging_level_for_all_handlers(logging.ERROR)
    # disable_stream_handlers_for_namespace("proteusPy")

    _logger.info("Starting DisulfideExtractor at time %s", datetime.datetime.now())

    mess = (
        f"proteusPy DisulfideExtractor v{__version__}\n"
        f"PDB model directory:       {PDB_DIR}\n"
        f"Data directory:            {DATA_DIR}\n"
        f"Module data directory:     {MODULE_DATA}\n"
        f"Repo data directory:       {REPO_DATA}\n"
        f"Number of .ent files:      {num_ent_files}\n"
        f"Cα cutoff:                 {args.cutoff}\n"
        f"Sγ cutoff:                 {args.gamma}\n"
        f"Extract:                   {args.extract}\n"
        f"Build:                     {args.build}\n"
        f"Update:                    {args.update}\n"
        f"Full:                      {args.full}\n"
        f"Subset:                    {args.subset}\n"
        f"Verbose:                   {args.verbose}\n"
        f"Threads:                   {args.threads}\n"
        f"Forge:                     {args.forge}\n"
        f"Environment:               {args.env}\n"
        f"Starting at:               {datetime.datetime.now()}\n"
    )

    _logger.info(mess)

    do_stuff(
        extract=args.extract,
        build=args.build,
        update=args.update,
        full=args.full,
        subset=args.subset,
        verbose=args.verbose,
        cutoff=args.cutoff,
        threads=args.threads,
        forge=args.forge,
        env=args.env,
        sg_cutoff=args.gamma,
    )

    end = time.time()

    elapsed_time = timedelta(seconds=end - start)

    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Processing completed in {int(hours)}h:{int(minutes)}m:{int(seconds)}")


if __name__ == "__main__":
    main()

    print(f"Finished DisulfideExtraction at time %s", datetime.datetime.now())


# End of file
