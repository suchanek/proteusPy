"""
Disulfide class analysis using `proteusPy.Disulfide` package. Disulfide binary families are defined
using the +/- formalism of Schmidt et al. (Biochem, 2006, 45, 7429-7433), across
all 32 possible classes ($$2^5$$). Classes are named per the paper's convention.

+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| IDX|   chi1_s |   chi2_s |   chi3_s |   chi4_s |   chi5_s |   class_id | SS_Classname   | FXN        |
+====+==========+==========+==========+==========+==========+============+================+============+
|  0 |       -1 |       -1 |       -1 |       -1 |       -1 |      00000 | -LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  1 |       -1 |       -1 |       -1 |       -1 |        1 |      00002 | 00002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  2 |       -1 |       -1 |       -1 |        1 |       -1 |      00020 | -LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  3 |       -1 |       -1 |       -1 |        1 |        1 |      00022 | 00022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  4 |       -1 |       -1 |        1 |       -1 |       -1 |      00200 | -RHStaple      | Allosteric |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  5 |       -1 |       -1 |        1 |       -1 |        1 |      00202 | 00202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  6 |       -1 |       -1 |        1 |        1 |       -1 |      00220 | 00220          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  7 |       -1 |       -1 |        1 |        1 |        1 |      00222 | 00222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  8 |       -1 |        1 |       -1 |       -1 |       -1 |      02000 | 02000          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  9 |       -1 |        1 |       -1 |       -1 |        1 |      02002 | 02002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 10 |       -1 |        1 |       -1 |        1 |       -1 |      02020 | -LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 11 |       -1 |        1 |       -1 |        1 |        1 |      02022 | 02022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 12 |       -1 |        1 |        1 |       -1 |       -1 |      02200 | -RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 13 |       -1 |        1 |        1 |       -1 |        1 |      02202 | 02202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 14 |       -1 |        1 |        1 |        1 |       -1 |      02220 | -RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 15 |       -1 |        1 |        1 |        1 |        1 |      02222 | 02222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 16 |        1 |       -1 |       -1 |       -1 |       -1 |      20000 | ±LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 17 |        1 |       -1 |       -1 |       -1 |        1 |      20002 | +LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 18 |        1 |       -1 |       -1 |        1 |       -1 |      20020 | ±LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 19 |        1 |       -1 |       -1 |        1 |        1 |      20022 | +LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 20 |        1 |       -1 |        1 |       -1 |       -1 |      20200 | ±RHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 21 |        1 |       -1 |        1 |       -1 |        1 |      20202 | +RHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 22 |        1 |       -1 |        1 |        1 |       -1 |      20220 | ±RHHook        | Catalytic  |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 23 |        1 |       -1 |        1 |        1 |        1 |      20222 | 20222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 24 |        1 |        1 |       -1 |       -1 |       -1 |      22000 | -/+LHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 25 |        1 |        1 |       -1 |       -1 |        1 |      22002 | 22002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 26 |        1 |        1 |       -1 |        1 |       -1 |      22020 | +/-LHStaple    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 27 |        1 |        1 |       -1 |        1 |        1 |      22022 | +LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 28 |        1 |        1 |        1 |       -1 |       -1 |      22200 | -/+RHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 29 |        1 |        1 |        1 |       -1 |        1 |      22202 | +RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 30 |        1 |        1 |        1 |        1 |       -1 |      22220 | ±RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 31 |        1 |        1 |        1 |        1 |        1 |      22222 | +RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+

The sextant class approach is unique to proteusPy, wherein the dihedral circle is divided into 6 quadrants,
and a dihedral angle vector defined by characterizing each dihedral angle into one of these six quadrants.
This yields $6^{5}$ or 7776 possible classes. This program analyzes the RCSB database and creates graphs illustrating
the membership across the binary and sextant classes. The graphs are stored in the global SAVE_DIR location.
Binary analysis takes approximately 28 minutes with Sextant analysis taking about
75 minutes on a 2023 M3 Max Macbook Pro.

Author: Eric G. Suchanek, PhD. Last Modified: 8/2/2024
"""

import argparse
import os
import pickle
import threading
import time
from datetime import timedelta
from typing import List

import numpy as np
from tqdm import tqdm

from proteusPy import Disulfide, DisulfideList, DisulfideLoader, Load_PDB_SS
from proteusPy.data import DATA_DIR

SAVE_DIR = "/Users/egs/Documents/proteusPy/"


from collections import deque

import pandas as pd


def task(
    loader: DisulfideLoader,
    six_or_bin: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    result_list: DisulfideList,
    pbar: tqdm,
    total_ss: int,
    cutoff: float,
    do_graph: bool,
    do_consensus: bool,
    save_dir: str,
    prefix: str,
    position: int,
    tasknum: int,
):
    """
    Processes a range of disulfide classes and updates the progress bar.

    :param loader: DisulfideLoader instance to load disulfides.
    :param six_or_bin: DataFrame containing class and disulfide IDs.
    :param start_idx: Starting index for processing.
    :param end_idx: Ending index for processing.
    :param result_list: List to store the resulting disulfides.
    :param pbar: tqdm progress bar for overall progress.
    :param total_ss: Total number of disulfides.
    :param cutoff: Cutoff percentage to filter classes.
    :param do_graph: Boolean flag to generate and save graphs.
    :param do_consensus: Boolean flag to generate consensus conformations.
    :param save_dir: Directory to save the output files.
    :param prefix: Prefix for the output file names.
    :param position: Vertical position of the progress bar.
    :param tasknum: Task number for identification.
    """
    task_pbar = tqdm(
        total=end_idx - start_idx,
        desc=f"--> task {tasknum+1:2}...",
        position=position + 1,
        leave=False,
        ncols=80,
    )

    for idx in range(start_idx, end_idx):
        row = six_or_bin.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        fname = os.path.join(save_dir, "classes", f"{prefix}_{cls}.png")

        class_disulfides_deque = deque()

        for ssid in ss_list:
            _ss = loader[ssid]
            class_disulfides_deque.append(_ss)
            task_pbar.update(1)

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

        pbar.update(1)


def analyze_six_classes(
    loader: DisulfideLoader,
    do_graph=False,
    do_consensus=True,
    cutoff=0.1,
    verbose=False,
) -> DisulfideList:
    """
    Analyze the six classes of disulfide bonds.

    :param loader: The disulfide loader object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param do_consensus: Whether or not to compute average conformations for each class. Default is True.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    _PBAR_COLS = 85

    class_filename = os.path.join(DATA_DIR, "SS_consensus_class_sext.pkl")
    six = loader.tclass.sixclass_df
    tot_classes = six.shape[0]
    res_list = DisulfideList([], "SS_6class_Avg_SS")
    total_ss = len(loader.SSList)

    pbar = tqdm(range(tot_classes), ncols=_PBAR_COLS)

    # loop over all rows
    for idx in pbar:
        row = six.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        fname = os.path.join(SAVE_DIR, "classes", f"ss_class_sext_{cls}.png")
        pbar.set_postfix({"CLS": cls, "Cnt": tot})  # update the progress bar

        class_disulfides_deque = deque()

        pbar2 = tqdm(ss_list, ncols=_PBAR_COLS, leave=False)
        for ssid in pbar2:
            _ss = loader[ssid]
            class_disulfides_deque.append(_ss)

        class_disulfides = list(class_disulfides_deque)
        sslist = DisulfideList(class_disulfides, cls, quiet=True)

        if do_graph:
            sslist.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            if verbose:
                print(
                    f"--> analyze_six_classes(): Computing avg conformation for: {cls}"
                )

            avg_conformation = sslist.Average_Conformation

            # build the average disulfide for the class
            ssname = f"{cls}_avg"
            exemplar = Disulfide(ssname)
            exemplar.build_model(
                avg_conformation[0],
                avg_conformation[1],
                avg_conformation[2],
                avg_conformation[3],
                avg_conformation[4],
            )
            res_list.append(exemplar)

    if do_consensus:
        print(
            f"--> analyze_six_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def analyze_six_classes_threaded(
    loader: DisulfideLoader,
    do_graph=False,
    do_consensus=False,
    cutoff=0.1,
    num_threads=8,
    verbose=False,
) -> DisulfideList:
    """
    Analyze the six classes of disulfide bonds.

    :param loader: The ``proteusPy.DisulfideLoader`` object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param do_consensus: Whether or not to compute average conformations for each class. Default is True.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.
    :param num_threads: Number of threads to use for processing. Default is 4.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    _PBAR_COLS = 85

    class_filename = os.path.join(DATA_DIR, "SS_consensus_class_sext.pkl")
    six = loader.tclass.sixclass_df
    tot_classes = six.shape[0]
    res_list = DisulfideList([], "SS_6class_Avg_SS")
    total_ss = len(loader.SSList)

    threads = []
    chunk_size = tot_classes // num_threads
    result_lists = [[] for _ in range(num_threads)]

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_threads - 1 else tot_classes
        pbar_index = 2 * i + 1  # so the task pbar is displayed in the correct position
        pbar = tqdm(
            total=end_idx - start_idx,
            desc=f"Thread {i+1:2}",
            position=pbar_index,
            leave=False,
        )
        thread = threading.Thread(
            target=task,
            args=(
                loader,
                six,
                start_idx,
                end_idx,
                result_lists[i],
                pbar,
                total_ss,
                cutoff,
                do_graph,
                do_consensus,
                SAVE_DIR,
                "ss_class_sext",
                pbar_index,
                i,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for result_list in result_lists:
        res_list.extend(result_list)

    if do_consensus:
        print(
            f"--> analyze_six_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def analyze_binary_classes_threaded(
    loader: DisulfideLoader,
    do_graph=False,
    do_consensus=False,
    cutoff=0.1,
    num_threads=4,
) -> DisulfideList:
    """
    Analyze the binary classes of disulfide bonds.

    :param loader: The ``proteusPy.DisulfideLoader`` object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param do_consensus: Whether or not to compute average conformations for each class. Default is True. The program writes the
                        consensus structures to a .pkl file in the packages ``data`` directory.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.
    :param num_threads: Number of threads to use for processing. Default is 4.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    _PBAR_COLS = 85

    class_filename = os.path.join(DATA_DIR, "SS_consensus_class32.pkl")

    bin = loader.tclass.classdf
    tot_classes = bin.shape[0]
    res_list = DisulfideList([], "SS_32class_Avg_SS")
    total_ss = len(loader.SSList)

    chunk_size = tot_classes // num_threads
    threads = []
    result_lists = [[] for _ in range(num_threads)]

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_threads - 1 else tot_classes
        pbar_index = 2 * i + 1  # so the task pbar is displayed in the correct position

        pbar = tqdm(
            total=end_idx - start_idx, desc=f"Thread {i+1:2}", position=i, leave=False
        )
        thread = threading.Thread(
            target=task,
            args=(
                loader,
                bin,
                start_idx,
                end_idx,
                result_lists[i],
                pbar,
                total_ss,
                cutoff,
                do_graph,
                do_consensus,
                SAVE_DIR,
                "ss_class_bin",
                pbar_index,
                i,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for result_list in result_lists:
        res_list.extend(result_list)

    if do_consensus:
        print(
            f"--> analyze_binary_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


# with 172,000 disulfides, this takes about 32 minutes, M3 Max Macbook Pro


def analyze_binary_classes(
    loader, do_graph=False, do_consensus=True, cutoff=0.1, verbose=False
) -> DisulfideList:
    """
    Analyze the binary classes of disulfide bonds.

    :param loader: The disulfide loader object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param do_consensus: Whether or not to compute average conformations for each class. Default is True.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    _PBAR_COLS = 85

    class_filename = f"{DATA_DIR}SS_consensus_class32.pkl"

    bin = loader.tclass.classdf
    tot_classes = bin.shape[0]
    res_list = DisulfideList([], "SS_32class_Avg_SS")
    total_ss = len(loader.SSList)

    pbar = tqdm(range(tot_classes), ncols=_PBAR_COLS)

    # loop over all rows
    for idx in pbar:
        row = bin.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        fname = f"{SAVE_DIR}classes/ss_class_bin_{cls}.png"
        pbar.set_postfix({"CLS": cls, "Cnt": tot})  # update the progress bar

        # class_disulfides = DisulfideList([], cls, quiet=True)
        class_disulfides_deque = deque()

        pbar2 = tqdm(ss_list, ncols=_PBAR_COLS, leave=False)
        for ssid in pbar2:
            _ss = loader[ssid]
            class_disulfides_deque.append(_ss)

        class_disulfides = list(class_disulfides_deque)
        sslist = DisulfideList(class_disulfides, cls, quiet=True)

        if do_graph:
            sslist.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            if verbose:
                print(
                    f"--> analyze_binary_classes(): Computing avg conformation for: {cls}"
                )
            avg_conformation = sslist.Average_Conformation

            # build the average disulfide for the class
            ssname = f"{cls}_avg"
            exemplar = Disulfide(ssname)
            exemplar.build_model(
                avg_conformation[0],
                avg_conformation[1],
                avg_conformation[2],
                avg_conformation[3],
                avg_conformation[4],
            )
            res_list.append(exemplar)

    if do_consensus:
        print(
            f"--> analyze_binary_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def plot_classes_vs_cutoff(PDB_SS: DisulfideLoader, cutoff, steps):
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


def analyze_classes(
    binary: bool,
    sextant: bool,
    all: bool,
    threads: int = 4,
    do_graph: bool = False,
    cutoff: float = 0.1,
    verbose: bool = False,
):
    # main program begins
    if all:
        analyze_six_classes_threaded(
            PDB_SS,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=cutoff,
            verbose=verbose,
            num_threads=threads,
        )

        analyze_binary_classes(
            PDB_SS, do_graph=do_graph, do_consensus=True, cutoff=cutoff, verbose=verbose
        )

        return

    if sextant:
        print("Analyzing sextant classes.")
        ss_classlist = analyze_six_classes_threaded(
            PDB_SS,
            do_graph=do_graph,
            do_consensus=True,
            cutoff=0.1,
            num_threads=threads,
        )

    if binary:
        ss_classlist = analyze_binary_classes(
            PDB_SS, do_graph=do_graph, do_consensus=True, cutoff=0.1, verbose=verbose
        )

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "-b",
    "--binary",
    help="Analyze binary classes.",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-s",
    "--sextant",
    help="Analyze sextant classes.",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-t",
    "--threads",
    type=int,
    help="Number of threads to use. NOT IMPLEMENTED YET.",
)
parser.add_argument(
    "-a",
    "--all",
    help="Both binary and sextant classes.",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "-g",
    "--graph",
    help="Create class graphs.",
    action=argparse.BooleanOptionalAction,
)

parser.set_defaults(binary=False)
parser.set_defaults(sextant=False)
parser.set_defaults(all=False)
parser.set_defaults(graph=False)
parser.set_defaults(threads=4)
parser.set_defaults(cutoff=0.1)


args = parser.parse_args()
sextant = args.sextant
binary = args.binary
all = args.all
threads = args.threads
do_graph = args.graph
cutoff = args.cutoff

print(f"Starting extraction with arguments: {args}")
print(
    f"Binary: {binary}, Sextant: {sextant}, All: {all}, Threads: {threads}, Cutoff: {cutoff}, Graph: {do_graph}"
)

start = time.time()
PDB_SS = Load_PDB_SS(verbose=True, subset=False)

analyze_classes(binary, sextant, all, threads=threads, do_graph=do_graph, cutoff=cutoff)

end = time.time()

elapsed = end - start

print(
    f"Disulfide Class Analysis Complete! \nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
)

# end of file
