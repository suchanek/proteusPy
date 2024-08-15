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
import pickle

# Last Modification: 2/19/2024
# Cα Cβ Sγ
import time
from datetime import timedelta

import numpy as np
from tqdm import tqdm

import proteusPy
from proteusPy import Disulfide, DisulfideList, Load_PDB_SS
from proteusPy.data import DATA_DIR

SAVE_DIR = "/Users/egs/Documents/proteusPy/"


def analyze_six_classes(
    loader, do_graph=True, do_consensus=True, cutoff=0.1
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

    class_filename = f"{DATA_DIR}SS_consensus_class_sext.pkl"

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

        fname = f"{SAVE_DIR}classes/ss_class_sext_{cls}.png"
        pbar.set_postfix({"CLS": cls, "Cnt": tot})  # update the progress bar

        class_disulfides = DisulfideList([], cls, quiet=True)

        pbar2 = tqdm(ss_list, ncols=_PBAR_COLS, leave=False, position=1)
        for ssid in pbar2:
            _ss = loader[ssid]
            class_disulfides.append(_ss)
            # remove it from the overall list to increase speed for searching
            # loader.SSList.remove(_ss)

        if do_graph:
            class_disulfides.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            # print(f'--> analyze_six_classes(): Computing avg conformation for: {cls}')
            avg_conformation = class_disulfides.Average_Conformation

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


def analyze_binary_classes(
    loader, do_graph=True, do_consensus=True, cutoff=0.1
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

        class_disulfides = DisulfideList([], cls, quiet=True)

        pbar2 = tqdm(ss_list, ncols=_PBAR_COLS, leave=False)
        for ssid in pbar2:
            _ss = loader[ssid]
            class_disulfides.append(_ss)
            # remove it from the overall list to increase speed for searching
            # loader.SSList.remove(_ss)

        if do_graph:
            class_disulfides.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            print(
                f"--> analyze_binary_classes(): Computing avg conformation for: {cls}"
            )
            avg_conformation = class_disulfides.Average_Conformation

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


def analyze_classes(binary: bool, sextant: bool, all: bool, graph: bool, cutoff: float):
    # main program begins
    if all:
        analyze_six_classes(PDB_SS, do_graph=graph, do_consensus=True, cutoff=cutoff)
        analyze_binary_classes(PDB_SS, do_graph=graph, do_consensus=True, cutoff=cutoff)
        return

    if sextant:
        # ss_classlist = DisulfideList([], 'PDB_SS_SIX_CLASSES')
        ss_classlist = analyze_six_classes(
            PDB_SS, do_graph=graph, do_consensus=True, cutoff=cutoff
        )

    if binary:
        # ss_classlist = DisulfideList([], 'PDB_SS_BINARY_CLASSES')
        ss_classlist = analyze_binary_classes(
            PDB_SS, do_graph=graph, do_consensus=True, cutoff=cutoff
        )

    return


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
        default=1,
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
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        help="Cutoff percentage for class filtering.",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()
    return args


PDB_SS = Load_PDB_SS(verbose=True, subset=False)


args = get_args()
sextant = args.sextant
binary = args.binary
all = args.all
graph = args.graph
cutoff = args.cutoff

start = time.time()
analyze_classes(binary, sextant, all, graph, cutoff)
end = time.time()

elapsed = end - start

print(
    f"Disulfide Class Analysis Complete! \nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
)

# end of file
