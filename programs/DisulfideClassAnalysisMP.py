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

Author: Eric G. Suchanek, PhD.
"""

import argparse
import multiprocessing
import os
import pickle
import time
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from proteusPy import Disulfide, DisulfideList, DisulfideLoader, Load_PDB_SS
from proteusPy.data import DATA_DIR

SAVE_DIR = "/Users/egs/Documents/proteusPy/"


def task(
    start_idx: int,
    end_idx: int,
    six: pd.DataFrame,  # Assuming six is a pandas DataFrame
    total_ss: int,
    cutoff: float,
    loader: DisulfideLoader,  # Replace 'LoaderType' with the actual type of loader
    do_graph: bool,
    do_consensus: bool,
    result_queue: multiprocessing.Queue,
    pbar: tqdm,
) -> None:
    result_list = []
    for idx in range(start_idx, end_idx):
        row = six.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        fname = os.path.join(SAVE_DIR, "classes", f"ss_class_six_{cls}.png")

        class_disulfides = DisulfideList([], cls, quiet=True)

        for ssid in ss_list:
            _ss = loader[ssid]
            class_disulfides.append(_ss)

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
    result_queue.put(result_list)
    print(f"Process {multiprocessing.current_process().name} finished processing.")


def analyze_six_classes(
    loader: DisulfideLoader, do_graph=True, do_consensus=True, cutoff=0.1, num_threads=8
) -> DisulfideList:
    """
    Analyze the six classes of disulfide bonds.

    :param loader: The ``proteusPy.DisulfideLoader`` object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param do_consensus: Whether or not to compute average conformations for each class. Default is True.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.
    :param num_threads: Number of threads to use for processing. Default is 8.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    _PBAR_COLS = 85

    class_filename = os.path.join(DATA_DIR, "SS_consensus_class_sext.pkl")
    six = loader.tclass.sixclass_df
    tot_classes = six.shape[0]
    res_list = DisulfideList([], "SS_6class_Avg_SS")
    total_ss = len(loader.SSList)

    chunk_size = tot_classes // num_threads
    processes = []
    result_queue = multiprocessing.Queue()

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_threads - 1 else tot_classes
        pbar = tqdm(
            total=end_idx - start_idx, desc=f"Process {i+1:2}", position=i, leave=False
        )
        process = multiprocessing.Process(
            target=task,
            args=(
                start_idx,
                end_idx,
                six,
                total_ss,
                cutoff,
                loader,
                do_graph,
                do_consensus,
                result_queue,
                pbar,
            ),
            name=f"Process-{i+1:2}",
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not result_queue.empty():
        res_list.extend(result_queue.get())

    if do_consensus:
        print(
            f"--> analyze_six_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def analyze_binary_classes(
    loader: DisulfideLoader, do_graph=True, do_consensus=True, cutoff=0.1, num_threads=8
) -> DisulfideList:
    _PBAR_COLS = 85

    class_filename = os.path.join(DATA_DIR, "SS_consensus_class32.pkl")
    binary = loader.tclass.binaryclass_df
    tot_classes = binary.shape[0]
    res_list = DisulfideList([], "SS_binary_Avg_SS")
    total_ss = len(loader.SSList)

    chunk_size = tot_classes // num_threads
    processes = []
    result_queue = multiprocessing.Queue()

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_threads - 1 else tot_classes
        pbar = tqdm(
            total=end_idx - start_idx, desc=f"Process {i+1:2}", position=i, leave=False
        )
        process = multiprocessing.Process(
            target=task,
            args=(start_idx, end_idx, result_queue, pbar),
            name=f"Process-{i+1:2}",
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not result_queue.empty():
        res_list.extend(result_queue.get())

    if do_consensus:
        print(
            f"--> analyze_binary_classes(): Writing consensus structures to: {class_filename}"
        )
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

    return res_list


def plot_classes_vs_cutoff(cutoff, steps):
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


def analyze_classes(binary: bool, sextant: bool, all: bool, threads: int = 4):
    # main program begins
    if all:
        analyze_six_classes(
            PDB_SS, do_graph=True, do_consensus=True, cutoff=0.0, num_threads=threads
        )
        analyze_binary_classes(
            PDB_SS, do_graph=True, do_consensus=True, cutoff=0.0, num_threads=threads
        )
        return

    if sextant:
        # ss_classlist = DisulfideList([], 'PDB_SS_SIX_CLASSES')
        ss_classlist = analyze_six_classes(
            PDB_SS, do_graph=True, do_consensus=True, cutoff=0.0, num_threads=threads
        )

    if binary:
        # ss_classlist = DisulfideList([], 'PDB_SS_BINARY_CLASSES')
        ss_classlist = analyze_binary_classes(
            PDB_SS, do_graph=True, do_consensus=True, cutoff=0.0, num_threads=threads
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
    help="Number of threads to use.",
)
parser.add_argument(
    "-a",
    "--all",
    help="Both binary and sextant classes.",
    action=argparse.BooleanOptionalAction,
)

parser.set_defaults(binary=False)
parser.set_defaults(sextant=False)
parser.set_defaults(all=False)

args = parser.parse_args()
sextant = args.sextant
binary = args.binary
all = args.all
threads = args.threads

start = time.time()
PDB_SS = Load_PDB_SS(verbose=True, subset=False)

analyze_classes(binary, sextant, all, threads=threads)
end = time.time()

elapsed = end - start

print(
    f"Disulfide Class Analysis Complete! \nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
)

# end of file
