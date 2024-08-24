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


def task(
    loader: DisulfideLoader,
    six_or_bin: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    total_ss: int,
    cutoff: float,
    do_graph: bool,
    do_consensus: bool,
    save_dir: str,
    prefix: str,
    position: int,
    tasknum: int,
    result_queue: multiprocessing.Queue,
):
    result_list = []

    for idx in range(start_idx, end_idx):

        row = six_or_bin.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        task_pbar = tqdm(
            total=len(ss_list),
            desc=f"{Fore.YELLOW}  Task {tasknum+1:2}{Style.RESET_ALL}".ljust(10),
            position=position + 1,
            leave=False,
            ncols=PBAR_COLS,
            bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.YELLOW, Style.RESET_ALL),
        )

        fname = os.path.join(save_dir, f"{prefix}_{cls}.png")

        class_disulfides_deque = deque()
        counter = 0
        update_freq = 50

        for ssid in ss_list:
            _ss = loader[ssid]
            class_disulfides_deque.append(_ss)
            counter += 1
            if counter % update_freq == 0 or counter == len(ss_list) - 1:
                task_pbar.update(update_freq)

        task_pbar.close()
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

    result_queue.put(result_list)
    return


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

    if do_sextant:
        class_filename = os.path.join(DATA_DIR, SS_CONSENSUS_FILE)
        SAVE_DIR = os.path.join(SAVE_DIR, "sextant")
        six_or_bin = loader.tclass.sixclass_df
        tot_classes = six_or_bin.shape[0]
        res_list = DisulfideList([], "SS_6class_Avg_SS")
        pix = sextant_classes_vs_cutoff(loader, cutoff)
        print(
            f"--> analyze_six_classes(): Expecting {pix} graphs for the sextant classes."
        )
    else:
        class_filename = os.path.join(DATA_DIR, SS_CONSENSUS_BIN_FILE)
        SAVE_DIR = os.path.join(SAVE_DIR, "binary")
        six_or_bin = loader.tclass.classdf
        tot_classes = six_or_bin.shape[0]
        res_list = DisulfideList([], "SS_32class_Avg_SS")

    total_ss = len(loader.SSList)

    chunk_size = tot_classes // num_processes
    result_queue = multiprocessing.Queue()

    # Create the overall progress bar
    overall_pbar = tqdm(
        total=tot_classes,
        desc=f"{Fore.GREEN}Overall Progress{Style.RESET_ALL}".ljust(20),
        position=0,
        leave=True,
        ncols=PBAR_COLS,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.GREEN, Style.RESET_ALL),
    )

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_processes - 1 else tot_classes
        pbar_index = 2 * i + 2

        process = multiprocessing.Process(
            target=task,
            args=(
                loader,
                six_or_bin,
                start_idx,
                end_idx,
                total_ss,
                cutoff,
                do_graph,
                do_consensus,
                SAVE_DIR,
                prefix,
                pbar_index,
                i,
                result_queue,
            ),
        )
        processes.append(process)
        process.start()

    res_list = DisulfideList([], "SS_Avg_SS")

    # Collect the results from each process
    for _ in range(num_processes):
        result_list = result_queue.get()
        res_list.extend(result_list)

    # Ensure all processes have finished
    for process in processes:
        process.join()

    # overall_pbar.close()

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
    import matplotlib.pyplot as plt

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

    PDB_SS = Load_PDB_SS(verbose=False, subset=False)
    PDB_SS.describe()

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
