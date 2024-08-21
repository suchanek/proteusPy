import argparse
import datetime
import glob
import logging
import os
import pickle
import sys
import time
import tkinter as tk
from pathlib import Path
from shutil import copy
from tkinter import messagebox, ttk

from tqdm import tqdm

from proteusPy import (
    Extract_Disulfides,
    Extract_Disulfides_From_List,
    set_logger_level_for_module,
)
from proteusPy.ProteusGlobals import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SS_PICKLE_FILE,
    SS_PROBLEM_SUBSET_ID_FILE,
    SS_SUBSET_DICT_PICKLE_FILE,
    SS_SUBSET_PICKLE_FILE,
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
__version__ = "1.0.2"


def do_extract(verbose, full, subset, cutoff, prune, progress_callback):
    from proteusPy import Extract_Disulfides, load_list_from_file

    if subset:
        if verbose:
            print("--> Extracting the SS subset...")

        for i in range(1000):
            Extract_Disulfides(
                numb=i,
                pdbdir=PDB_DIR,
                datadir=DATA_DIR,
                picklefile=SS_SUBSET_PICKLE_FILE,
                problemfile=SS_PROBLEM_SUBSET_ID_FILE,
                verbose=verbose,
                quiet=True,
                dist_cutoff=cutoff,
                prune=prune,
            )
            progress_callback(i + 1, 1000)

    if full:
        if verbose:
            print("--> Extracting the SS full dataset. This will take a while...")

        for i in range(num_ent_files):
            Extract_Disulfides(
                numb=i,
                verbose=verbose,
                quiet=True,
                pdbdir=PDB_DIR,
                datadir=DATA_DIR,
                dist_cutoff=cutoff,
                prune=prune,
            )
            progress_callback(i + 1, num_ent_files)
    return


def do_build(verbose, full, subset, cutoff):
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
    prune=True,
    progress_callback=None,
):
    _extract = extract
    _build = build
    _full = full
    _update = update
    _subset = subset
    _verbose = verbose

    if all:
        _extract = _build = _update = _subset = _full = True

    if _extract:
        if verbose:
            print(f"Extracting with cutoff: {cutoff}")
        do_extract(_verbose, _full, _subset, cutoff, prune, progress_callback)

    if _build:
        if verbose:
            print(f"Building:")
        do_build(_verbose, _full, _subset, cutoff)

    if _update:
        if verbose:
            print(f"Copying: {DATA_DIR} to {REPO_DATA}")

        update_repo(DATA_DIR)

    return


def start_process():
    all = all_var.get()
    extract = extract_var.get()
    build = build_var.get()
    full = full_var.get()
    update = update_var.get()
    subset = subset_var.get()
    verbose = verbose_var.get()
    cutoff = float(cutoff_var.get())

    start = time.time()
    set_logger_level_for_module("proteusPy", logging.ERROR)

    print(
        f"proteusPy DisulfideExtractor v{__version__}\n"
        f"PDB model directory:       {PDB_DIR}\n"
        f"Data directory:            {DATA_DIR}\n"
        f"Module data directory:     {MODULE_DATA}\n"
        f"Repo data directory:       {REPO_DATA}\n"
        f"Number of .ent files:      {num_ent_files}\n"
        f"Using cutoff:              {cutoff}\n"
        f"Extract:                   {extract}\n"
        f"Build:                     {build}\n"
        f"Update:                    {update}\n"
        f"Full:                      {full}\n"
        f"Subset:                    {subset}\n"
        f"Verbose:                   {verbose}\n"
        f"Starting at:               {datetime.datetime.now()}\n"
    )

    def progress_callback(current, total):
        progress_var.set((current / total) * 100)
        root.update_idletasks()

    do_stuff(
        all=all,
        extract=extract,
        build=build,
        update=update,
        full=full,
        subset=subset,
        verbose=verbose,
        cutoff=cutoff,
        progress_callback=progress_callback,
    )

    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds")
    messagebox.showinfo(
        "Process Completed", f"Processing completed in {end - start:.2f} seconds"
    )


def main():
    global all_var, extract_var, build_var, full_var, update_var, subset_var, verbose_var, cutoff_var, progress_var, root

    root = tk.Tk()
    root.title("Disulfide Extractor")

    all_var = tk.BooleanVar()
    extract_var = tk.BooleanVar()
    build_var = tk.BooleanVar()
    full_var = tk.BooleanVar()
    update_var = tk.BooleanVar()
    subset_var = tk.BooleanVar()
    verbose_var = tk.BooleanVar()
    cutoff_var = tk.StringVar(value="-1.0")
    progress_var = tk.DoubleVar()

    tk.Checkbutton(root, text="All", variable=all_var).grid(row=0, sticky=tk.W)
    tk.Checkbutton(root, text="Extract", variable=extract_var).grid(row=1, sticky=tk.W)
    tk.Checkbutton(root, text="Build", variable=build_var).grid(row=2, sticky=tk.W)
    tk.Checkbutton(root, text="Full", variable=full_var).grid(row=3, sticky=tk.W)
    tk.Checkbutton(root, text="Update", variable=update_var).grid(row=4, sticky=tk.W)
    tk.Checkbutton(root, text="Subset", variable=subset_var).grid(row=5, sticky=tk.W)
    tk.Checkbutton(root, text="Verbose", variable=verbose_var).grid(row=6, sticky=tk.W)

    tk.Label(root, text="Cutoff:").grid(row=7, sticky=tk.W)
    tk.Entry(root, textvariable=cutoff_var).grid(row=7, column=1)

    tk.Button(root, text="Start", command=start_process).grid(row=8, columnspan=2)

    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=12, columnspan=2, sticky=tk.W + tk.E)

    root.mainloop()


if __name__ == "__main__":
    main()

# End of file
