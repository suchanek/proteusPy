import argparse
import os
import pickle
import shutil
import time
from datetime import timedelta
from multiprocessing import Manager, Process
from pathlib import Path

import numpy as np
from colorama import init
from tqdm import tqdm

import proteusPy as pp
from proteusPy import CA_CUTOFF, SG_CUTOFF, SS_CONSENSUS_BIN_FILE, SS_CONSENSUS_OCT_FILE

HOME_DIR = Path.home()
PDB = Path(os.getenv("PDB", HOME_DIR / "pdb"))
DATA_DIR = PDB / "data"
SAVE_DIR = HOME_DIR / "Documents" / "proteusPyDocs" / "classes"
REPO_DIR = HOME_DIR / "repos" / "proteusPy" / "data"
OCTANT = SAVE_DIR / "octant"
SEXTANT = SAVE_DIR / "sextant"
BINARY = SAVE_DIR / "binary"
OCTANT.mkdir(parents=True, exist_ok=True)
SEXTANT.mkdir(parents=True, exist_ok=True)
BINARY.mkdir(parents=True, exist_ok=True)
PBAR_COLS = 78
init(autoreset=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--binary", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "-o", "--octant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("-t", "--threads", type=int, default=8)
    parser.add_argument(
        "-g", "--graph", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("-c", "--cutoff", type=float, default=0.0)
    parser.add_argument(
        "-v", "--verbose", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "-u", "--update", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def task(
    eight_or_bin,
    start_idx,
    end_idx,
    result_list,
    total_ss,
    cutoff,
    do_graph,
    save_dir,
    prefix,
):
    loader = pp.Load_PDB_SS(
        verbose=True, subset=False, cutoff=CA_CUTOFF, sg_cutoff=SG_CUTOFF
    )

    for idx in range(start_idx, end_idx):
        row = eight_or_bin.iloc[idx]
        cls = row["class_id"]
        ss_list = row["ss_id"]
        tot = len(ss_list)
        if 100 * tot / total_ss < cutoff:
            continue

        class_disulfides_array = np.empty(len(ss_list), dtype=object)
        for idx, ssid in enumerate(ss_list):
            _ss = loader[ssid]
            if _ss is None:
                continue

            try:
                class_disulfides_array[idx] = loader[ssid]
            except KeyError:
                continue

        class_disulfides = pp.DisulfideList(
            list(class_disulfides_array), cls, quiet=True, fast=True
        )

        if do_graph:
            fname = Path(save_dir) / f"{prefix}_{cutoff}_{cls}_{len(ss_list)}.png"
            class_disulfides.display_torsion_statistics(
                display=False, save=True, fname=fname, light=True, stats=False
            )

        avg_conformation = class_disulfides.average_conformation
        exemplar = pp.Disulfide(f"{cls}", torsions=avg_conformation)
        result_list.append(exemplar)


def analyze_classes_multiprocessing(
    do_graph=False,
    cutoff=0.0,
    num_processes=8,
    verbose=True,
    do_octant=True,
    prefix="ss",
):
    loader = pp.Load_PDB_SS(
        verbose=verbose, subset=False, cutoff=CA_CUTOFF, sg_cutoff=SG_CUTOFF
    )
    save_dir = OCTANT if do_octant else BINARY
    eight_or_bin = loader.tclass.eightclass_df if do_octant else loader.tclass.classdf
    tot_classes = eight_or_bin.shape[0]
    total_ss = len(loader.SSList)
    chunk_size = tot_classes // num_processes

    del loader

    manager = Manager()
    result_lists = manager.list()

    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_processes - 1 else tot_classes
        process = Process(
            target=task,
            args=(
                eight_or_bin,
                start_idx,
                end_idx,
                result_lists,
                total_ss,
                cutoff,
                do_graph,
                save_dir,
                prefix,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    res_list = pp.DisulfideList(
        [], "SS_8class_Avg_SS" if do_octant else "SS_32class_Avg_SS"
    )
    for result_list in result_lists:
        res_list.extend(result_list)

    class_filename = Path(DATA_DIR) / (
        SS_CONSENSUS_OCT_FILE if do_octant else SS_CONSENSUS_BIN_FILE
    )
    with open(class_filename, "wb+") as f:
        pickle.dump(res_list, f)

    return res_list


def analyze_classes(binary, octant, processes=4, do_graph=False, cutoff=0.0):
    if octant:
        print("Analyzing octant classes.")
        analyze_classes_multiprocessing(do_graph, cutoff, processes, True, "ss_oct")

    if binary:
        print("Analyzing binary classes.")
        analyze_classes_multiprocessing(do_graph, cutoff, processes, False, "ss_bin")


def update_repository(source_dir, repo_dir, binary=False, octant=False):
    if binary:
        source = Path(source_dir) / SS_CONSENSUS_BIN_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_BIN_FILE
        shutil.copy(source, dest)

    if octant:
        source = Path(source_dir) / SS_CONSENSUS_OCT_FILE
        dest = Path(repo_dir) / SS_CONSENSUS_OCT_FILE
        shutil.copy(source, dest)


def main():
    args = get_args()

    analyze_classes(
        binary=args.binary,
        octant=args.octant,
        processes=args.threads,
        do_graph=args.graph,
        cutoff=args.cutoff,
    )

    if args.update:
        print("Updating repository with consensus classes.")
        update_repository(DATA_DIR, REPO_DIR, binary=args.binary, octant=args.octant)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print(
        f"\nDisulfide Class Analysis Complete!\nElapsed time: {timedelta(seconds=elapsed)} (h:m:s)"
    )
