import argparse
import multiprocessing
import os
import pickle
import time
from collections import deque
from datetime import timedelta

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tqdm import tqdm

from proteusPy.ProteusGlobals import SS_CONSENSUS_BIN_FILE, SS_CONSENSUS_FILE

# Initialize colorama
init(autoreset=True)

from proteusPy import Disulfide, DisulfideList, DisulfideLoader, Load_PDB_SS

HOME = os.getenv("HOME")
PDB = os.getenv("PDB")
if PDB is None:
    PDB = os.path.join(HOME, "pdb")

DATA_DIR = os.path.join(PDB, "data")
SAVE_DIR = os.path.join(HOME, "Documents", "proteusPyDocs", "classes")
REPO_DIR = os.path.join(HOME, "repos", "proteusPy", "data")

PBAR_COLS = 78

# Step 1: Define a global variable for the loader
global_loader = None


# Step 2: Create an initializer function to initialize the global loader
def init_worker(loader):
    global global_loader
    global_loader = loader
    print(f"{Fore.GREEN}Worker initialized.{Style.RESET_ALL}")


def task(args):
    (
        six_or_bin_flag,
        start_idx,
        end_idx,
        cutoff,
        do_graph,
        do_consensus,
        save_dir,
        prefix,
        position,
        tasknum,
    ) = args

    loader = global_loader
    result_list = []
    total_ss = len(loader.SSList)
    six_or_bin = loader.tclass.sixclass_df if six_or_bin_flag else loader.tclass.classdf
    tot_classes = six_or_bin.shape[0]

    overall_pbar = tqdm(
        total=end_idx - start_idx,
        desc=f"Process {tasknum+1:2}".ljust(10),
        position=position,
        leave=False,
        ncols=PBAR_COLS,
    )
    for idx in range(start_idx, end_idx):
        row = six_or_bin.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        task_pbar = tqdm(
            total=tot,
            desc=f"  subtask {tasknum+1:2}".ljust(10),
            position=position + 1,
            leave=False,
            ncols=PBAR_COLS,
        )

        fname = os.path.join(save_dir, f"{prefix}_{cls}.png")

        class_disulfides_deque = deque()
        counter = 0
        update_freq = 10

        for ssid in ss_list:
            _ss = loader[ssid]
            class_disulfides_deque.append(_ss)
            counter += 1
            if counter % update_freq == 0 or counter == len(ss_list) - 1:
                task_pbar.update(update_freq)

        class_disulfides = DisulfideList(list(class_disulfides_deque), cls, quiet=True)
        if do_graph:
            class_disulfides.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        if do_consensus:
            avg_conformation = class_disulfides.Average_Conformation

            ssname = f"{cls}_avg"
            exemplar = Disulfide(ssname)
            exemplar.build_model(
                avg_conformation[0],
                avg_conformation[1],
                avg_conformation[2],
                avg_conformation[3],
                avg_conformation[4],
            )
            result_list.append(exemplar)

        overall_pbar.set_postfix({"CLS": cls})
        overall_pbar.update(1)

    task_pbar.close()
    overall_pbar.close()

    return result_list


def analyze_classes_multiprocess(
    loader: DisulfideLoader,
    do_graph=False,
    do_consensus=False,
    cutoff=0.0,
    num_processes=6,
    verbose=False,
    do_sextant=True,
    prefix="ss_class",
) -> DisulfideList:

    global SAVE_DIR

    six_or_bin = loader.tclass.sixclass_df if do_sextant else loader.tclass.classdf
    tot_classes = six_or_bin.shape[0]
    chunk_size = tot_classes // num_processes

    if do_sextant:
        class_filename = os.path.join(DATA_DIR, SS_CONSENSUS_FILE)
        SAVE_DIR = os.path.join(SAVE_DIR, "sextant")
        six_or_bin_flag = True
        res_list = DisulfideList([], "SS_6class_Avg_SS")
        pix = sextant_classes_vs_cutoff(loader, cutoff)
        print(
            f"--> analyze_six_classes(): Expecting {pix} graphs for the sextant classes."
        )
    else:
        class_filename = os.path.join(DATA_DIR, SS_CONSENSUS_BIN_FILE)
        SAVE_DIR = os.path.join(SAVE_DIR, "binary")
        six_or_bin = False
        res_list = DisulfideList([], "SS_32class_Avg_SS")

    pool_args = [
        (
            do_sextant,
            i * chunk_size,
            (i + 1) * chunk_size if i != num_processes - 1 else tot_classes,
            cutoff,
            do_graph,
            do_consensus,
            SAVE_DIR,
            prefix,
            2 * i,
            i,
        )
        for i in range(num_processes)
    ]

    with multiprocessing.Pool(
        processes=num_processes, initializer=init_worker, initargs=(loader,)
    ) as pool:
        results = pool.map(task, pool_args)

    res_list = DisulfideList([], "SS_Avg_SS")
    for result in results:
        res_list.extend(result)

    if do_consensus:
        print(f"Writing consensus structures to: {class_filename}")
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def analyze_classes(
    loader: DisulfideLoader,
    binary: bool,
    sextant: bool,
    all: bool,
    processes: int = 8,
    do_graph: bool = False,
    cutoff: float = 0.0,
    verbose: bool = False,
):
    if all:
        analyze_classes_multiprocess(
            loader,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=cutoff,
            verbose=verbose,
            num_processes=processes,
            do_sextant=True,
        )

        analyze_classes_multiprocess(
            loader,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=0,
            verbose=verbose,
            num_processes=processes,
            do_sextant=False,
        )

        return

    if sextant:
        print("Analyzing sextant classes.")
        analyze_classes_multiprocess(
            loader,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=cutoff,
            verbose=verbose,
            num_processes=processes,
            do_sextant=True,
            prefix="ss_class_sext",
        )

    if binary:
        analyze_classes_multiprocess(
            loader,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=0.0,
            verbose=verbose,
            num_processes=processes,
            do_sextant=False,
            prefix="ss_class_bin",
        )

    return


def plot_sixclass_vs_cutoff(PDB_SS: DisulfideLoader, cutoff, steps, verbose=False):
    """
    Plot the total percentage and number of members for each class against the cutoff value.

    :param cutoff: Percent cutoff value for filtering the classes.
    :return: None
    """
    import matplotlib.pyplot as plt

    _cutoff = np.linspace(0, cutoff, steps)
    tot_list = []
    members_list = []

    for c in _cutoff:
        class_df = PDB_SS.tclass.filter_sixclass_by_percentage(c)
        tot = class_df["percentage"].sum()
        tot_list.append(tot)
        members_list.append(class_df.shape[0])
        if verbose:
            print(
                f"Cutoff: {c:5.3} accounts for {tot:7.2f}% and is {class_df.shape[0]:5} members long."
            )

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(_cutoff, tot_list, label="Total percentage", color="blue")
    ax2.plot(_cutoff, members_list, label="Number of members", color="red")

    ax1.set_xlabel("Cutoff")
    ax1.set_ylabel("Total percentage", color="blue")
    ax2.set_ylabel("Number of members", color="red")

    plt.show()


def sextant_classes_vs_cutoff(loader: DisulfideLoader, cutoff):
    """
    Return number of members for the sextant class for a given cutoff value.

    :param cutoff: Percent cutoff value for filtering the classes.
    :return: None
    """

    class_df = loader.tclass.filter_sixclass_by_percentage(cutoff)
    return class_df.shape[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--binary",
        help="Analyze binary classes.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-s",
        "--sextant",
        help="Analyze sextant classes.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use. NOT IMPLEMENTED YET.",
        default=8,
    )
    parser.add_argument(
        "-a",
        "--all",
        help="Both binary and sextant classes.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-g",
        "--graph",
        help="Create class graphs.",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        help="Cutoff percentage for class filtering.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose output.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-u",
        "--update",
        help="Update repository with the consensus classes.",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    return args


def Update_Repository(source_dir, repo_dir, verbose=True, binary=False, sextant=False):
    """Copy the consensus classes to the repository."""
    import shutil

    if binary:
        source = os.path.join(source_dir, SS_CONSENSUS_BIN_FILE)
        dest = os.path.join(repo_dir, SS_CONSENSUS_BIN_FILE)

        if verbose:
            print(f"Copying {source} to {dest}")

        shutil.copy(source, dest)

    if sextant:
        source = os.path.join(source_dir, SS_CONSENSUS_FILE)
        dest = os.path.join(repo_dir, SS_CONSENSUS_FILE)

        if verbose:
            print(f"Copying {source} to {dest}")

        shutil.copy(source, dest)

    return


def main():
    args = get_args()
    sextant = args.sextant
    binary = args.binary
    all = args.all
    threads = args.threads
    do_graph = args.graph
    cutoff = args.cutoff
    do_update = args.update

    # Clear the terminal window
    print("\033c", end="")

    print("Starting Disulfide Class analysis with arguments:")
    print(
        f"Binary:                {binary}\n"
        f"Sextant:               {sextant}\n"
        f"All:                   {all}\n"
        f"Threads:               {threads}\n"
        f"Cutoff:                {cutoff}\n"
        f"Graph:                 {do_graph}\n"
        f"Consensus:             True \n"
        f"Update:                {do_update}\n"
        f"Verbose:               {args.verbose}\n"
        f"Data directory:        {DATA_DIR}\n"
        f"Save directory:        {SAVE_DIR}\n"
        f"Repository directory:  {REPO_DIR}\n"
        f"Home directory:        {HOME}\n"
        f"PDB directory:         {PDB}\n"
        f"Loading PDB SS data...\n"
    )

    PDB_SS = Load_PDB_SS(verbose=True, subset=False)

    analyze_classes(
        PDB_SS,
        binary,
        sextant,
        all,
        processes=threads,
        do_graph=do_graph,
        cutoff=cutoff,
    )

    if do_update:
        print("Updating repository with consensus classes.")
        Update_Repository(DATA_DIR, REPO_DIR, binary=binary, sextant=sextant)


if __name__ == "__main__":
    start = time.time()

    main()

    end = time.time()
    elapsed = end - start

    print(
        f"\n----------------------\nDisulfide Class Analysis Complete! \nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
    )

# end of file
