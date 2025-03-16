# pylint: disable=C0301
# pylint: disable=C0103
# Last modification: 2025-03-16 15:08:03 -egs-

"""
Disulfide class consensus structure extraction using `proteusPy.Disulfide` package.
Disulfide binary families are defined using the +/- formalism of Schmidt et al.
(Biochem, 2006, 45, 7429-7433), across all 32 possible classes ($$2^5$$). Classes
are named per the paper's convention.

|   class_id | SS_Classname   | FXN        |   count |   incidence |   percentage |
|-----------:|:---------------|:-----------|--------:|------------:|-------------:|
|      00000 | -LHSpiral      | UNK        |   40943 |  0.23359    |    23.359    |
|      00002 | 00002          | UNK        |    9391 |  0.0535781  |     5.35781  |
|      00020 | -LHHook        | UNK        |    4844 |  0.0276363  |     2.76363  |
|      00022 | 00022          | UNK        |    2426 |  0.0138409  |     1.38409  |
|      00200 | -RHStaple      | Allosteric |   16146 |  0.092117   |     9.2117   |
|      00202 | 00202          | UNK        |    1396 |  0.00796454 |     0.796454 |
|      00220 | 00220          | UNK        |    7238 |  0.0412946  |     4.12946  |
|      00222 | 00222          | UNK        |    6658 |  0.0379856  |     3.79856  |
|      02000 | 02000          | UNK        |    7104 |  0.0405301  |     4.05301  |
|      02002 | 02002          | UNK        |    8044 |  0.0458931  |     4.58931  |
|      02020 | -LHStaple      | UNK        |    3154 |  0.0179944  |     1.79944  |
|      02022 | 02022          | UNK        |    1146 |  0.00653822 |     0.653822 |
|      02200 | -RHHook        | UNK        |    7115 |  0.0405929  |     4.05929  |
|      02202 | 02202          | UNK        |    1021 |  0.00582507 |     0.582507 |
|      02220 | -RHSpiral      | UNK        |    8989 |  0.0512845  |     5.12845  |
|      02222 | 02222          | UNK        |    7641 |  0.0435939  |     4.35939  |
|      20000 | ±LHSpiral      | UNK        |    5007 |  0.0285662  |     2.85662  |
|      20002 | +LHSpiral      | UNK        |    1611 |  0.00919117 |     0.919117 |
|      20020 | ±LHHook        | UNK        |    1258 |  0.00717721 |     0.717721 |
|      20022 | +LHHook        | UNK        |     823 |  0.00469542 |     0.469542 |
|      20200 | ±RHStaple      | UNK        |     745 |  0.00425042 |     0.425042 |
|      20202 | +RHStaple      | UNK        |     538 |  0.00306943 |     0.306943 |
|      20220 | ±RHHook        | Catalytic  |    1907 |  0.0108799  |     1.08799  |
|      20222 | 20222          | UNK        |    1159 |  0.00661239 |     0.661239 |
|      22000 | -/+LHHook      | UNK        |    3652 |  0.0208356  |     2.08356  |
|      22002 | 22002          | UNK        |    2052 |  0.0117072  |     1.17072  |
|      22020 | ±LHStaple      | UNK        |    1791 |  0.0102181  |     1.02181  |
|      22022 | +LHStaple      | UNK        |     579 |  0.00330334 |     0.330334 |
|      22200 | -/+RHHook      | UNK        |    8169 |  0.0466062  |     4.66062  |
|      22202 | +RHHook        | UNK        |     895 |  0.0051062  |     0.51062  |
|      22220 | ±RHSpiral      | UNK        |    3581 |  0.0204305  |     2.04305  |
|      22222 | +RHSpiral      | UNK        |    8254 |  0.0470912  |     4.70912  |

The octant class approach is unique to ``proteusPy``, wherein the dihedral circle
for the dihedral angles X1-X5 is divided into 8 sections and then represented as
a string of length 5, (class id) defined by characterizing each dihedral angle
into one of these sections. This yields $8^{5}$ or 32,768 possible classes. This
program analyzes the RCSB database and creates graphs illustrating the membership
across the binary and octant classes. The graphs are stored in the global SAVE_DIR
location.

Initial release: 1/12/2024
Binary analysis takes approximately 20 minutes with octant analysis taking
about 75 minutes on a 2023 M3 Max Macbook Pro. (single-threaded).

Update 8/28/2024 - multithreading is implemented and runs well up to around 10
threads on a 2023 M3 Max Macbook Pro. Octant analysis takes around 22 minutes with
6 threads. Binary analysis takes around 25 minutes with 6 threads.

Update 2/22/25 - after re-writing the DisulfideLoader class to use an index-based
approach to class selection, the octant and binary class analysis runs in about 2
minutes with 14 threads!


Author: Eric G. Suchanek, PhD. Last Modified: 2025-01-06 19:29:49
"""

import argparse
import os
import pickle
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tqdm import tqdm

from proteusPy import (
    SS_CONSENSUS_BIN_FILE,
    SS_CONSENSUS_OCT_FILE,
    Disulfide,
    DisulfideClassManager,
    DisulfideList,
    DisulfideLoader,
    configure_master_logger,
    create_logger,
    set_logger_level_for_module,
)

# Constants
HOME_DIR: Path = Path.home()
PDB: Path = Path(os.getenv("PDB", HOME_DIR / "pdb"))
DATA_DIR: Path = PDB / "data"
SAVE_DIR: Path = HOME_DIR / "Documents" / "proteusPyDocs" / "classes"
MODULE_DIR: Path = HOME_DIR / "repos" / "proteusPy" / "proteusPy" / "data"
REPO_DIR: Path = HOME_DIR / "repos" / "proteusPy" / "data"
OCTANT: Path = SAVE_DIR / "octant"
SEXTANT: Path = SAVE_DIR / "sextant"
BINARY: Path = SAVE_DIR / "binary"
MINIFORGE_DIR: Path = HOME_DIR / Path("miniforge3/envs")
MAMBAFORGE_DIR: Path = HOME_DIR / Path("mambaforge/envs")
VENV_DIR: Path = Path("lib/python3.12/site-packages/proteusPy/data")
PBAR_COLS: int = 78

# Directory Setup
OCTANT.mkdir(parents=True, exist_ok=True)
SEXTANT.mkdir(parents=True, exist_ok=True)
BINARY.mkdir(parents=True, exist_ok=True)

PERCENTILE: float = -1  # percentile cutoff for Sg and Ca filtering.
# Initialize colorama
init(autoreset=True)

_logger = create_logger("__name__")
configure_master_logger("DisulfidClass_Analysis.log")
set_logger_level_for_module("proteusPy", "ERROR")


def get_args() -> argparse.Namespace:
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

    :return: An object containing the parsed command-line arguments.
    :rtype:
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
        "-p",
        "--percentile",
        help="Cutoff percentage for building the database",
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
    return parser.parse_args()


def task(
    loader: DisulfideLoader,
    overall_pbar: tqdm,
    start_idx: int,
    end_idx: int,
    result_list: DisulfideList,
    metrics_list: List[Dict],
    pbar: tqdm,
    cutoff: float,
    do_graph: bool,
    save_dir: str,
    prefix: str,
    base: int,
    classlist: List[str],
) -> None:
    """
    Processes a range of lines in the disulfide class dict for the binary or octant disulfide classes.

    :param loader: DisulfideLoader instance to load disulfides
    :type loader: DisulfideLoader
    :param overall_pbar: Progress bar for overall progress
    :type overall_pbar: tqdm
    :param start_idx: Starting index for processing
    :type start_idx: int
    :param end_idx: Ending index for processing
    :type end_idx: int
    :param result_list: List to store the resulting disulfides
    :type result_list: DisulfideList
    :param metrics_list: List to store metrics for each class
    :type metrics_list: List[Dict]
    :param pbar: Progress bar for thread-specific progress
    :type pbar: tqdm
    :param cutoff: Cutoff percentage to filter classes (0.04 is good)
    :type cutoff: float
    :param do_graph: Boolean flag to generate and save graphs
    :type do_graph: bool
    :param save_dir: Directory to save the output files
    :type save_dir: str
    :param prefix: Prefix for the output file names
    :type prefix: str
    :param base: Base value for the class type, 2 for binary and 8 for octant
    :type base: int
    :param classlist: List of class names
    :type classlist: List[str]
    """
    # pylint: disable=W0212

    try:
        for idx in range(start_idx, end_idx):
            cls = classlist[idx]
            tot_class_ss = len(loader._class_indices_from_tors_df(cls, base))
            percentage = 100 * tot_class_ss / loader.TotalDisulfides

            if percentage < cutoff:
                pbar.set_postfix({"SKP": cls})
                pbar.update(1)
                overall_pbar.update(1)
                continue

            pbar.set_postfix({"CLS": cls})
            class_disulfides = loader.sslist_from_class(cls, base=base, cutoff=cutoff)
            pbar.update(1)

            sub = "o" if base == 8 else "b"

            fname = Path(save_dir) / f"{cls}{sub}_{cutoff}_{tot_class_ss}.png"

            if do_graph:
                class_disulfides.display_torsion_statistics(
                    display=False,
                    save=True,
                    fname=fname,
                    theme="light",
                )

            # Calculate metrics
            avg_ca_distance = class_disulfides.average_ca_distance
            avg_energy = class_disulfides.average_energy

            # Calculate standard deviations
            ca_distances = [ss.ca_distance for ss in class_disulfides.data]
            energies = [ss.energy for ss in class_disulfides.data]

            std_ca_distance = np.std(ca_distances) if ca_distances else 0.0
            std_energy = np.std(energies) if energies else 0.0

            class_str = DisulfideClassManager.class_to_binary(cls, base=base)
            tor_vals, _ = class_disulfides.calculate_torsion_statistics()

            tor_mean_vals = tor_vals.loc["mean"]
            tor_std_vals = tor_vals.loc["std"]

            # Store metrics
            metrics = {
                "class": cls,
                "class_str": class_str,
                "count": tot_class_ss,
                "percentage": percentage,
                "avg_energy": avg_energy,
                "std_energy": std_energy,
                "avg_ca_distance": avg_ca_distance,
                "std_ca_distance": std_ca_distance,
                "chi1_mean": tor_mean_vals[0],
                "chi1_std": tor_std_vals[0],
                "chi2_mean": tor_mean_vals[1],
                "chi2_std": tor_std_vals[1],
                "chi3_mean": tor_mean_vals[2],
                "chi3_std": tor_std_vals[2],
                "chi4_mean": tor_mean_vals[3],
                "chi4_std": tor_std_vals[3],
                "chi5_mean": tor_mean_vals[4],
                "chi5_std": tor_std_vals[4],
            }
            metrics_list.append(metrics)

            avg_conformation = class_disulfides.average_conformation
            ssname = f"{cls}b" if base == 2 else f"{cls}o"
            exemplar = Disulfide(ssname, torsions=avg_conformation)
            result_list.append(exemplar)
            overall_pbar.update(1)
    finally:
        pbar.close()


def analyze_classes_threaded(
    loader: DisulfideLoader,
    do_graph: bool = False,
    cutoff: float = 0.0,
    num_threads: int = 8,
    do_octant: bool = True,
    prefix: str = "ss",
) -> Tuple[DisulfideList, pd.DataFrame]:
    """
    Analyze the classes of disulfide bonds using multithreading.

    :param loader: The DisulfideLoader object
    :type loader: DisulfideLoader
    :param do_graph: Whether to display torsion statistics graphs
    :type do_graph: bool
    :param cutoff: The cutoff percentage for each class
    :type cutoff: float
    :param num_threads: Number of threads to use for processing
    :type num_threads: int
    :param do_octant: Whether to analyze octant classes (True) or binary (False)
    :type do_octant: bool
    :param prefix: Prefix for output filenames
    :type prefix: str
    :return: Tuple containing list of disulfide bonds representing average conformations and DataFrame with metrics
    :rtype: Tuple[DisulfideList, pd.DataFrame]
    """
    save_dir = None
    tors_df = loader.TorsionDF
    class_filename = None
    metrics_filename = None
    eight_or_bin = None
    res_list = None
    pix = None
    base = None
    classlist = None

    if do_octant:
        class_filename = Path(DATA_DIR) / SS_CONSENSUS_OCT_FILE
        metrics_filename = Path(OCTANT) / f"octant_class_metrics_{cutoff:.2f}.csv"
        save_dir = OCTANT
        base = 8
        eight_or_bin = loader.tclass.eightclass_df
        res_list = DisulfideList([], f"SS_32class_Avg_SS_{cutoff:.2f}")
        pix = loader.classes_vs_cutoff(cutoff, base=base)
        classlist = tors_df["octant_class_string"].unique()

        if do_graph:
            print(f"Expecting {pix} graphs for the octant classes.")
    else:
        class_filename = Path(DATA_DIR) / SS_CONSENSUS_BIN_FILE
        metrics_filename = Path(BINARY) / f"binary_class_metrics_{cutoff:.2f}.csv"
        save_dir = BINARY
        eight_or_bin = loader.tclass.binaryclass_df
        total_classes = eight_or_bin.shape[0]  # 32
        res_list = DisulfideList([], "SS_32class_Avg_SS")
        pix = 32
        base = 2
        classlist = tors_df["binary_class_string"].unique()

    total_classes = len(classlist)
    threads = []
    chunk_size = total_classes // num_threads
    result_lists = [DisulfideList([]) for _ in range(num_threads)]
    metrics_lists = [[] for _ in range(num_threads)]

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
        pbar_index = i + 1
        pbar = tqdm(
            total=end_idx - start_idx,
            desc=f"{Fore.BLUE}Thread {i+1:2}{Style.RESET_ALL}".ljust(10),
            position=pbar_index,
            leave=False,
            ncols=PBAR_COLS + 10,
            bar_format="{l_bar}%s{bar}{r_bar}%s" % (Fore.BLUE, Style.RESET_ALL),
            miniters=100,
        )
        thread = threading.Thread(
            target=task,
            args=(
                loader,
                overall_pbar,
                start_idx,
                end_idx,
                result_lists[i],
                metrics_lists[i],
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

    for thread in threads:
        thread.join()

    overall_pbar.close()

    # Combine results from all threads
    for result_list in result_lists:
        res_list.extend(result_list)

    # Combine metrics from all threads
    all_metrics = []
    for metrics_list in metrics_lists:
        all_metrics.extend(metrics_list)

    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(all_metrics)

    # Sort by percentage (descending)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="class", ascending=True)

    try:
        print(f"Writing consensus structures to: {class_filename}")
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)

        print(f"Writing class metrics to: {metrics_filename}")
        metrics_df.to_csv(metrics_filename, index=False)
    except IOError as e:
        _logger.error("Failed to write output files: %s", e)
        raise

    return res_list, metrics_df


def analyze_classes(
    loader: DisulfideLoader,
    binary: bool,
    octant: bool,
    threads: int = 4,
    do_graph: bool = False,
    cutoff: float = 0.0,
) -> None:
    """
    Analyzes disulfide bond classes using the provided loader.

    :param loader: The DisulfideLoader instance used to load and process disulfide bonds
    :type loader: DisulfideLoader
    :param binary: If True, analyzes binary classes
    :type binary: bool
    :param octant: If True, analyzes octant classes
    :type octant: bool
    :param threads: The number of threads to use for analysis
    :type threads: int
    :param do_graph: If True, generates graphs for the analysis
    :type do_graph: bool
    :param cutoff: The cutoff value for filtering disulfides
    :type cutoff: float
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


def convert_class_to_string(cls: str) -> str:
    """
    Converts a binary class representation to a string where both '0' is converted to '-' and '1'
    is converted to '+'.

    :param cls: The binary class string to convert
    :type cls: str
    :return: The converted string with '-' for each character
    :rtype: str
    """
    class_str = ""
    for char in cls:
        if char == "0":
            class_str += "-"
        else:
            class_str += "+"
    return class_str


def update_repository(
    source_dir: str,
    repo_dir: str,
    verbose: bool = True,
    binary: bool = False,
    octant: bool = False,
) -> None:
    """
    Copy the consensus class structures to the repository.

    :param source_dir: Source directory containing consensus files
    :type source_dir: str
    :param repo_dir: Destination repository directory
    :type repo_dir: str
    :param verbose: If True, print copy operations
    :type verbose: bool
    :param binary: If True, update binary consensus classes
    :type binary: bool
    :param octant: If True, update octant consensus classes
    :type octant: bool
    """
    if binary:
        source = Path(source_dir) / SS_CONSENSUS_BIN_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_BIN_FILE
        if verbose:
            print(f"Copying binary consensus classes from: {source} to {dest}")
        try:
            shutil.copy(source, dest)
        except IOError as e:
            _logger.error("Failed to copy binary consensus: %s", e)

    if octant:
        source = Path(source_dir) / SS_CONSENSUS_OCT_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_OCT_FILE
        if verbose:
            print(f"Copying octant consensus structures from {source} to {dest}")
        try:
            shutil.copy(source, dest)
        except IOError as e:
            _logger.error("Failed to copy octant consensus: %s", e)


def main() -> None:
    """
    Main function to execute the disulfide class consensus class extraction.
    """
    args = get_args()
    octant: bool = args.octant
    binary: bool = args.binary
    threads: int = args.threads
    do_graph: bool = args.graph
    cutoff: float = args.cutoff
    do_update: bool = args.update
    verbose: bool = args.verbose
    forge: str = args.forge
    env: str = args.env
    percentile = args.percentile

    if threads < 1:
        raise ValueError("Number of threads must be positive")

    print("\033c", end="")
    print("Starting Disulfide Class analysis with arguments:")
    print(
        f"Binary:                {binary}\n"
        f"Octant:                {octant}\n"
        f"Threads:               {threads}\n"
        f"Cutoff:                {cutoff}%\n"
        f"Percentile:            {percentile}\n"
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
        f"Loading PDB SS data..."
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

    try:
        pdb_ss = DisulfideLoader(
            verbose=verbose,
            subset=False,
            cutoff=-1,
            sg_cutoff=-1,
            percentile=percentile,
        )
        analyze_classes(
            pdb_ss,
            binary,
            octant,
            threads=threads,
            do_graph=do_graph,
            cutoff=cutoff,
        )
    except Exception as e:
        _logger.error("Analysis failed: %s", e)
        raise


if __name__ == "__main__":
    start: float = time.time()
    try:
        main()
    finally:
        end: float = time.time()
        elapsed: float = end - start
        print(
            f"\n----------------------\nDisulfide Class Analysis Complete!"
            f"\nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
        )

# EOF
