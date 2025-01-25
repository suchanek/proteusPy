import argparse
import logging
import os
import pickle
import threading
from pathlib import Path
from threading import Lock

import panel as pn
from colorama import init

import proteusPy as pp
from proteusPy import SS_CONSENSUS_BIN_FILE, SS_CONSENSUS_OCT_FILE

# Initialize Panel extension
pn.extension()

# Define directories
HOME_DIR = Path.home()
PDB = Path(os.getenv("PDB", HOME_DIR / "pdb"))
DATA_DIR = PDB / "data"
SAVE_DIR = HOME_DIR / "Documents" / "proteusPyDocs" / "classes"

OCTANT = SAVE_DIR / "octant"
BINARY = SAVE_DIR / "binary"
OCTANT.mkdir(parents=True, exist_ok=True)
BINARY.mkdir(parents=True, exist_ok=True)

_logger = pp.create_logger("DisulfideClass_Analysis", log_level=logging.INFO)

# Initialize colorama
init(autoreset=True)

# Global lock and counters
lock = Lock()
completed_tasks = 0


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--binary", action="store_true", help="Analyze binary classes."
    )
    parser.add_argument(
        "-o", "--octant", action="store_true", help="Analyze octant classes."
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=8, help="Number of threads to use."
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=-1.0,
        help="Cutoff percentage for class filtering.",
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="Create class graphs."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    return parser.parse_args()


args = get_args()
nthreads = args.threads

# Dashboard components
overall_progress = pn.indicators.Progress(
    name="Overall Progress", bar_color="success", width=400, value=0
)
overall_annotation = pn.widgets.StaticText(name="Overall Progress", value="0/0")

thread_progress_bars = [
    pn.indicators.Progress(name=f"Thread {i+1}", bar_color="info", width=400, value=0)
    for i in range(nthreads)
]
thread_annotations = [
    pn.widgets.StaticText(name=f"Thread {i+1}", value="0/0") for i in range(nthreads)
]

dashboards = pn.Column(
    "# Disulfide Class Analysis",
    pn.Row(overall_progress, overall_annotation),
    pn.Column(
        *[
            pn.Row(bar, annotation)
            for bar, annotation in zip(thread_progress_bars, thread_annotations)
        ]
    ),
)


def update_progress(progress_widget, annotation_widget, total, value):
    """
    Update the progress bar and annotation.

    :param progress_widget: Progress bar widget to update.
    :param annotation_widget: Annotation widget to update.
    :param total: Total number of tasks.
    :param value: Current progress value.
    """
    progress_widget.value = int(100 * value / total)
    annotation_widget.value = f"{value}/{total}"


import time


def task(
    loader,
    start_idx,
    end_idx,
    total,
    thread_idx,
    result_list,
    cutoff,
    do_graph,
    save_dir,
    prefix,
    base,
    classlist,
):
    global completed_tasks
    global lock
    local_completed = 0
    last_update_time = time.time()  # Track last update

    for idx in range(start_idx, end_idx):
        cls = classlist[idx]
        tot_class_ss = len(loader.class_indices_from_tors_df(cls, base))

        if 100 * tot_class_ss / loader.TotalDisulfides < cutoff:
            local_completed += 1
            with lock:
                completed_tasks += 1

                # Throttle UI updates to once every 200ms
                if time.time() - last_update_time > 0.2:
                    update_progress(
                        overall_progress, overall_annotation, total, completed_tasks
                    )
                    update_progress(
                        thread_progress_bars[thread_idx],
                        thread_annotations[thread_idx],
                        end_idx - start_idx,
                        local_completed,
                    )
                    last_update_time = time.time()
            continue

        # Process valid classes
        class_disulfides = loader.sslist_from_class(cls, base=base, cutoff=cutoff)
        if do_graph:
            fname = Path(save_dir) / f"{prefix}_{cutoff}_{cls}_{tot_class_ss}.png"
            class_disulfides.display_torsion_statistics(
                display=False, save=True, fname=fname, theme="light"
            )

        avg_conformation = class_disulfides.average_conformation
        ssname = f"{cls}"
        exemplar = pp.Disulfide(ssname, torsions=avg_conformation)
        result_list.append(exemplar)

        local_completed += 1
        with lock:
            completed_tasks += 1

            # Throttle UI updates to once every 200ms
            if time.time() - last_update_time > 0.2:
                update_progress(
                    overall_progress, overall_annotation, total, completed_tasks
                )
                update_progress(
                    thread_progress_bars[thread_idx],
                    thread_annotations[thread_idx],
                    end_idx - start_idx,
                    local_completed,
                )
                last_update_time = time.time()

    # Final thread update
    with lock:
        update_progress(
            thread_progress_bars[thread_idx],
            thread_annotations[thread_idx],
            end_idx - start_idx,
            local_completed,
        )
    pn.io.push_notebook(dashboards)  # Force UI synchronization
    return


def analyze_classes_threaded(loader, do_graph, cutoff, num_threads, do_octant, prefix):
    """
    Analyze classes using multiple threads.

    :param loader: DisulfideLoader instance.
    :param do_graph: Whether to generate graphs.
    :param cutoff: Cutoff percentage for filtering.
    :param num_threads: Number of threads to use.
    :param do_octant: Whether to analyze octant classes.
    :param prefix: Prefix for file names.
    :return: List of analyzed disulfides.
    """
    global completed_tasks

    save_dir = OCTANT if do_octant else BINARY
    tors_df = loader.TorsionDF
    classlist = (
        tors_df["octant_class_string"].unique()
        if do_octant
        else tors_df["binary_class_string"].unique()
    )
    total_classes = len(classlist)

    completed_tasks = 0
    update_progress(
        overall_progress, overall_annotation, total_classes, completed_tasks
    )

    chunk_size = (total_classes + num_threads - 1) // num_threads
    result_lists = [[] for _ in range(num_threads)]

    threads = []
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_classes)
        if start_idx < total_classes:
            athread = threading.Thread(
                target=task,
                args=(
                    loader,
                    start_idx,
                    end_idx,
                    total_classes,
                    i,
                    result_lists[i],
                    cutoff,
                    do_graph,
                    save_dir,
                    prefix,
                    8 if do_octant else 2,
                    classlist,
                ),
            )

            threads.append(athread)
            athread.start()

    for thread in threads:
        thread.join()

    res_list = pp.DisulfideList([], f"SS_32class_Avg_SS_{cutoff:.2f}")
    for result_list in result_lists:
        res_list.extend(result_list)

    class_filename = Path(DATA_DIR) / (
        SS_CONSENSUS_OCT_FILE if do_octant else SS_CONSENSUS_BIN_FILE
    )
    with open(class_filename, "wb+") as f:
        pickle.dump(res_list, f)

    assert (
        completed_tasks == total_classes
    ), f"Expected {total_classes}, but got {completed_tasks}"
    return res_list


def main():
    """Main entry point for the program."""
    pdb_ss = pp.DisulfideLoader(
        verbose=args.verbose, subset=False, cutoff=-1, sg_cutoff=-1
    )
    # pdb_ss.describe()

    if args.octant:
        analyze_classes_threaded(
            pdb_ss,
            do_graph=args.graph,
            cutoff=args.cutoff,
            num_threads=args.threads,
            do_octant=True,
            prefix="ss_oct",
        )
    if args.binary:
        analyze_classes_threaded(
            pdb_ss,
            do_graph=args.graph,
            cutoff=args.cutoff,
            num_threads=args.threads,
            do_octant=False,
            prefix="ss_bin",
        )
    print("Analysis complete.")


if __name__ == "__main__":
    # pn.state.add_periodic_callback(lambda: None, period=10)  # Keep the UI interactive
    server = pn.serve(
        dashboards, start=True, show=True, threaded=True, theme="dark-minimal"
    )
    main()
    server.stop()
