# pylint: disable=C0301
# pylint: disable=C0103
# Last modification: 2025-01-23 14:21:21 -egs-

"""
Disulfide class consensus structure extraction using `proteusPy.Disulfide` package. Disulfide
binary families are defined using the +/- formalism of Schmidt et al. (Biochem, 2006, 45, 
7429-7433), across all 32 possible classes ($$2^5$$). Classes are named per the paper's convention.

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

The octant class approach is unique to ``proteusPy``, wherein the dihedral circle for the dihedral angles X1-X5 
is divided into 8 sections, and a dihedral angle five-dimensional string, (class id) defined by characterizing each dihedral 
angle into one of these sections. This yields $8^{5}$ or 32,768 possible classes. This program analyzes the RCSB database 
and creates graphs illustrating the membership across the binary and octant classes. The graphs are stored in the 
global SAVE_DIR location. Binary analysis takes approximately 20 minutes with octant analysis taking about
75 minutes on a 2023 M3 Max Macbook Pro. (single-threaded).

Update 8/28/2024 - multithreading is implemented and runs well up to around 10 threads on a 2023 M3 Max Macbook Pro.
octant analysis takes around 22 minutes with 6 threads. Binary analysis takes around 25 minutes with 6 threads.

Update 1/24/2025 - rewrote to use index-based retrieval of classes from the torsion dataframe. This is much faster
than the previous method of using the class string to retrieve the classes. T
The analyses now take around 2 minutes to run!

Author: Eric G. Suchanek, PhD. Last Modified: 2025-01-24 10:47:00
"""

# plyint: disable=C0103

import argparse
import logging
import os
import pickle
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path

from colorama import Fore, Style, init
from tqdm import tqdm

import proteusPy as pp
from proteusPy import CA_CUTOFF, SG_CUTOFF, SS_CONSENSUS_BIN_FILE, SS_CONSENSUS_OCT_FILE

HOME_DIR = Path.home()
PDB = Path(os.getenv("PDB", HOME_DIR / "pdb"))

DATA_DIR = PDB / "data"
SAVE_DIR = HOME_DIR / "Documents" / "proteusPyDocs" / "classes"
MODULE_DIR = HOME_DIR / "repos" / "proteusPy" / "proteusPy" / "data"
REPO_DIR = HOME_DIR / "repos" / "proteusPy" / "data"

OCTANT = SAVE_DIR / "octant"
OCTANT.mkdir(parents=True, exist_ok=True)

SEXTANT = SAVE_DIR / "sextant"
SEXTANT.mkdir(parents=True, exist_ok=True)

BINARY = SAVE_DIR / "binary"
BINARY.mkdir(parents=True, exist_ok=True)

MINIFORGE_DIR = HOME_DIR / Path("miniforge3/envs")
MAMBAFORGE_DIR = HOME_DIR / Path("mambaforge/envs")

VENV_DIR = Path("lib/python3.12/site-packages/proteusPy/data")

PBAR_COLS = 78

_logger = pp.create_logger("DisulfideClass_Analysis", log_level=logging.INFO)

# pp.set_logger_level("proteusPy.DisulfideLoader", "INFO")

pp.configure_master_logger(
    "DisulfideClass_Analysis", enable_file_logging=True, log_level=logging.WARNING
)

# Initialize colorama
init(autoreset=True)


def get_args():
    """
    Parses and returns command-line arguments for the DisulfideClass_Analysis script.

    The following arguments are supported:
    - `-b`, `--binary`: Analyze binary classes (default: False).
    - `-o`, `--octant`: Analyze octant classes (default: False).
    - `-t`, `--threads`: Number of threads to use (default: 10).
    - `-f`, `--forge`: Forge directory (default: "miniforge3").
    - `-e`, `--env`: ProteusPy environment (default: "ppydev").
    - `-g`, `--graph`: Create class graphs (default: False).
    - `-c`, `--cutoff`: Cutoff percentage for class filtering (default: -1.0).
    - `-v`, `--verbose`: Enable verbose output (default: False).
    - `-u`, `--update`: Update repository with the consensus classes (default: True).

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--binary",
        help="Analyze binary classes.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-o",
        "--octant",
        help="Analyze octant classes.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use.",
        default=8,
    )
    parser.add_argument(
        "-f",
        "--forge",
        help="Forge directory.",
        type=str,
        default="miniforge3",
    )
    parser.add_argument(
        "-e",
        "--env",
        help="ProteusPy environment.",
        type=str,
        default="ppydev",
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
        default=-1.0,
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
        default=False,
    )

    args = parser.parse_args()
    return args


# task definition
def task(
    loader: pp.DisulfideLoader,
    overall_pbar: tqdm,
    start_idx: int,
    end_idx: int,
    result_list: pp.DisulfideList,
    pbar: tqdm,
    cutoff: float,
    do_graph: bool,
    save_dir: str,
    prefix: str,
    base: int,
    classlist: list,
):
    """
    Processes a range of lines in the disulfide class dict for the binary or octant
    disulfide classes and updates the progress bar.

    :param loader: DisulfideLoader instance to load disulfides.
    :param eight_or_bin: DataFrame containing class and disulfide IDs.
    :param start_idx: Starting index for processing.
    :param end_idx: Ending index for processing.
    :param result_list: List to store the resulting disulfides.
    :param pbar: tqdm progress bar for overall progress.
    :param cutoff: Cutoff percentage to filter classes.
    :param do_graph: Boolean flag to generate and save graphs.
    :param save_dir: Directory to save the output files.
    :param prefix: Prefix for the output file names.
    :param base: Base value for the class indices, 2 for binary and 8 for octant.
    :param classlist: List of class names.
    :return: None
    """

    for idx in range(start_idx, end_idx):
        cls = classlist[idx]
        tot_class_ss = len(loader.class_indices_from_tors_df(cls, base))

        # tot = loader.TotalDisulfides
        if 100 * tot_class_ss / loader.TotalDisulfides < cutoff:
            pbar.set_postfix({"SKP": cls})
            pbar.update(1)
            overall_pbar.update(1)
            continue

        pbar.set_postfix({"CLS": cls})

        class_disulfides = loader.sslist_from_class(cls, base=base, cutoff=cutoff)

        pbar.update(1)

        fname = Path(save_dir) / f"{prefix}_{cutoff}_{cls}_{tot_class_ss}.png"

        if do_graph:
            class_disulfides.display_torsion_statistics(
                display=False,
                save=True,
                fname=fname,
                theme="light",
            )

        avg_conformation = class_disulfides.average_conformation

        ssname = f"{cls}"
        exemplar = pp.Disulfide(ssname, torsions=avg_conformation)
        result_list.append(exemplar)

        overall_pbar.update(1)

    pbar.close()
    return


def analyze_classes_threaded(
    loader: pp.DisulfideLoader,
    do_graph=False,
    cutoff=0.0,
    num_threads=8,
    do_octant=True,
    prefix="ss",
) -> pp.DisulfideList:
    """
    Analyze the classes of disulfide bonds.

    :param loader: The ``proteusPy.DisulfideLoader`` object.
    :param do_graph: Whether or not to display torsion statistics graphs. Default is True.
    :param cutoff: The cutoff percentage for each class. If the percentage of disulfides for a class is below
                   this value, the class will be skipped. Default is 0.1.
    :param num_threads: Number of threads to use for processing. Default is 8.

    :return: A list of disulfide bonds, where each disulfide bond represents the average conformation for a class.
    """

    save_dir = None
    tors_df = loader.TorsionDF

    if do_octant:
        class_filename = Path(DATA_DIR) / SS_CONSENSUS_OCT_FILE
        save_dir = OCTANT
        eight_or_bin = loader.tclass.eightclass_df
        res_list = pp.DisulfideList([], f"SS_32class_Avg_SS_{cutoff:.2f}")
        pix = octant_classes_vs_cutoff(loader, cutoff)
        base = 8
        classlist = tors_df["octant_class_string"].unique()
        total_classes = len(classlist)

        if do_graph:
            print(f"Expecting {pix} graphs for the octant classes.")
    else:
        class_filename = Path(DATA_DIR) / SS_CONSENSUS_BIN_FILE
        save_dir = BINARY
        eight_or_bin = loader.tclass.binaryclass_df
        total_classes = eight_or_bin.shape[0]  # 32
        res_list = pp.DisulfideList([], "SS_32class_Avg_SS")
        pix = 32
        base = 2
        classlist = tors_df["binary_class_string"].unique()
        total_classes = len(classlist)

    threads = []
    chunk_size = total_classes // num_threads
    result_lists = [[] for _ in range(num_threads)]

    # Create the overall progress bar
    overall_pbar = tqdm(
        total=total_classes,
        desc=f"{Fore.GREEN}Overall Progress{Style.RESET_ALL}".ljust(20),
        position=0,
        leave=True,
        ncols=PBAR_COLS + 10,
        bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.GREEN, Style.RESET_ALL),
    )

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_threads - 1 else total_classes
        pbar_index = i + 1  # so the task pbar is displayed in the correct position
        pbar = tqdm(
            total=end_idx - start_idx,
            desc=f"{Fore.BLUE}Thread {i+1:2}{Style.RESET_ALL}".ljust(10),
            position=pbar_index,
            leave=False,
            ncols=PBAR_COLS + 10,
            bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.BLUE, Style.RESET_ALL),
            mininterval=1.0,
        )
        thread = threading.Thread(
            target=task,
            args=(
                loader,
                overall_pbar,
                start_idx,
                end_idx,
                result_lists[i],
                pbar,
                cutoff,
                do_graph,
                save_dir,
                prefix,
                base,
                classlist,
            ),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    overall_pbar.close()
    del overall_pbar

    # Combine the results from all threads, yielding the final list of consensus structures.
    for result_list in result_lists:
        res_list.extend(result_list)

    print(f"Writing consensus structures to: {class_filename}")
    with open(class_filename, "wb+") as f:
        pickle.dump(res_list, f)

    return res_list


def analyze_classes(
    loader: pp.DisulfideLoader,
    binary: bool,
    octant: bool,
    threads: int = 4,
    do_graph: bool = False,
    cutoff: float = 0.0,
):
    """
    Analyzes disulfide bond classes using the provided loader.

    This function can analyze binary classes, octant classes, or both, depending on the parameters.
    It uses threading to parallelize the analysis and can optionally generate graphs.

    :param loader: The DisulfideLoader instance used to load and process disulfide bonds.
    :type loader: DisulfideLoader
    :param binary: If True, analyzes binary classes.
    :type binary: bool
    :param octant: If True, analyzes octant classes.
    :type octant: bool
    :param threads: The number of threads to use for analysis. Default is 4.
    :type threads: int
    :param do_graph: If True, generates graphs for the analysis. Default is False.
    :type do_graph: bool
    :param cutoff: The cutoff value for filtering disulfides. Default is 0.0.
    :type cutoff: float
    :param verbose: If True, enables verbose output. Default is False.
    :type verbose: bool
    :return: None
    """

    if octant:
        print("Analyzing octant classes.")
        analyze_classes_threaded(
            loader,
            do_graph=do_graph,
            cutoff=cutoff,
            num_threads=threads,
            do_octant=True,
            prefix="ss_oct",
        )

    if binary:
        print("Analyzing binary classes.")

        analyze_classes_threaded(
            loader,
            do_graph=do_graph,
            cutoff=0.0,
            num_threads=threads,
            do_octant=False,
            prefix="ss_bin",
        )

    return


def octant_classes_vs_cutoff(loader: pp.DisulfideLoader, cutoff):
    """
    Return number of members for the octant class for a given cutoff value.

    :param cutoff: Percent cutoff value for filtering the classes.
    :return: None
    """

    class_df = loader.tclass.filter_class_by_percentage(cutoff, 8)
    return class_df.shape[0]


def update_repository(source_dir, repo_dir, verbose=True, binary=False, octant=False):
    """Copy the consensus classes to the repository."""

    if binary:
        source = Path(source_dir) / SS_CONSENSUS_BIN_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_BIN_FILE

        if verbose:
            print(f"Copying binary consensus classes from: {source} to {dest}")

        shutil.copy(source, dest)

    if octant:
        source = Path(source_dir) / SS_CONSENSUS_OCT_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_OCT_FILE

        if verbose:
            print(f"Copying octant consensus structures from {source} to {dest}")

        shutil.copy(source, dest)


def main():
    """
    Main function to execute the disulfide class consensus class extraction.

    This function parses command-line arguments and performs the analysis of disulfide bond classes.
    It can analyze binary classes, octant classes, or both, depending on the arguments provided.
    The function also supports generating graphs and updating consensus structures.

    :return: None
    """
    args = get_args()
    octant = args.octant
    binary = args.binary
    threads = args.threads
    do_graph = args.graph
    cutoff = args.cutoff
    do_update = args.update
    verbose = args.verbose
    forge = args.forge
    env = args.env

    # Clear the terminal window
    print("\033c", end="")

    print("Starting Disulfide Class analysis with arguments:")
    print(
        f"Binary:                {binary}\n"
        f"Octant:                {octant}\n"
        f"Threads:               {threads}\n"
        f"Cutoff:                {cutoff}%\n"
        f"Graph:                 {do_graph}\n"
        f"Update:                {do_update}\n"
        f"Verbose:               {verbose}\n"
        f"Data directory:        {DATA_DIR}\n"
        f"Save directory:        {SAVE_DIR}\n"
        f"Repository directory:  {REPO_DIR}\n"
        f"Home directory:        {HOME_DIR}\n"
        f"PDB directory:         {PDB}\n"
        f"Forge:                 {forge}\n"
        f"Env:                   {env}\n"
    )

    if do_update:
        print("Updating repository with consensus classes.")

        update_repository(DATA_DIR, REPO_DIR, binary=binary, octant=octant)
        update_repository(DATA_DIR, MODULE_DIR, binary=binary, octant=octant)

        if forge == "miniforge3":
            venv_dir = MINIFORGE_DIR / env / VENV_DIR
        else:
            venv_dir = MAMBAFORGE_DIR / env / VENV_DIR

        if verbose:
            print(f"Updating environment SS class files from: {DATA_DIR} to {venv_dir}")

        update_repository(DATA_DIR, venv_dir, binary=binary, octant=octant)
        return

    # pdb_ss = pp.Load_PDB_SS(
    #    verbose=verbose, subset=False, cutoff=CA_CUTOFF, sg_cutoff=SG_CUTOFF
    # )

    print("Building DisulfideLoader. Please be patient...")
    pdb_ss = pp.DisulfideLoader(verbose=verbose, subset=False, cutoff=-1, sg_cutoff=-1)

    analyze_classes(
        pdb_ss,
        binary,
        octant,
        threads=threads,
        do_graph=do_graph,
        cutoff=cutoff,
    )


if __name__ == "__main__":
    start = time.time()
    main()

    end = time.time()

    elapsed = end - start

    print(
        f"\n----------------------\nDisulfide Class Analysis Complete!"
        f"\nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
    )

# end of file
