#
# Program loads all .ent files in PDB_DIR, checks for SS_BOND
# consistency, and moves good files to the PDB_DIR/good directory,
# moves them to PDB_DIR/bad otherwise.
# Author: Eric G. Suchanek, PhD
# Last modification 7/24/2024


import logging
import os
import shutil
import sys
import time
from datetime import timedelta
from glob import glob

from tqdm import tqdm

from proteusPy import check_header_from_file

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
HOME_DIR = os.path.expanduser("~")
PDB_BASE = os.getenv("PDB")

if not os.path.isdir(PDB_BASE):
    logging.error(f"Error: The directory {PDB_BASE} does not exist.")
    sys.exit(1)
else:
    print(f"Found PDB directory at: {PDB_BASE}  ")

GOOD_DIR = os.path.join(PDB_BASE, "good/")
if not os.path.isdir(GOOD_DIR):
    logging.error(f"Error: The directory {GOOD_DIR} does not exist.")
    sys.exit(1)


BAD_DIR = os.path.join(PDB_BASE, "bad/")
if not os.path.isdir(BAD_DIR):
    logging.error(f"Error: The directory {BAD_DIR} does not exist.")
    sys.exit(1)


def extract_pdb_id(filename: str) -> str:
    """
    Extracts the PDB ID from a filename formatted as 'pdb{id}.ent'.

    Parameters:
    - filename (str): The filename to extract the ID from.

    Returns:
    - str: The extracted PDB ID.
    """
    if filename.startswith("pdb") and filename.endswith(".ent"):
        return filename[3:-4]
    else:
        raise ValueError(
            "Filename {filename} does not follow the expected format 'pdb{id}.ent'"
        )


def check_files(
    pdb_dir: str, good_dir: str, bad_dir: str, verbose=False, quiet=True
) -> None:
    """
    Checks all PDB files in the directory `pdb_dir` for SS bond consistency.

    Parameters:
    - pdb_dir (str): The directory containing the PDB files to check.
    - good_dir (str): The directory to move good files to.
    - bad_dir (str): The directory to move bad files to.
    """
    from proteusPy import load_disulfides_from_id

    def name_to_id(fname: str) -> str:
        """
        Returns the PDB ID from the filename.

        :param fname: Complete PDB filename
        :return: PDB ID
        """
        ent = fname[3:-4]
        return ent

    os.chdir(pdb_dir)
    all_pdb_files = glob("*.ent")

    if len(all_pdb_files) == 0:
        print(f"No PDB files! Exiting...")
        return None

    badcount = 0
    count = 0

    pbar = tqdm(all_pdb_files, ncols=80)
    for fname in pbar:
        entry = name_to_id(fname)

        if check_header_from_file(fname, verbose=verbose) > 0:
            badcount += 1
            destination_path = os.path.join(bad_dir, os.path.basename(fname))

            if os.path.exists(destination_path):
                os.remove(destination_path)

            shutil.move(fname, destination_path)
            print(f"Bad file: {fname} moved to {bad_dir}")
        else:
            sslist = load_disulfides_from_id(
                entry, verbose=verbose, quiet=quiet, pdb_dir=pdb_dir
            )
            # sslist = Extract_Disulfide(entry)
            if sslist is None or len(sslist) == 0:
                badcount += 1
                destination_path = os.path.join(bad_dir, os.path.basename(fname))

                # Check if the destination file exists and remove it if it does
                if os.path.exists(destination_path):
                    os.remove(destination_path)

                shutil.move(fname, destination_path)
                if verbose:
                    logging.info(f"Bad file: {fname} moved to {bad_dir}")
            else:
                destination_path = os.path.join(good_dir, os.path.basename(fname))

                if os.path.exists(destination_path):
                    os.remove(destination_path)

                shutil.move(fname, destination_path)
                if verbose:
                    logging.info(f"Good file: {fname} moved to {good_dir}")

        count += 1
        pbar.set_postfix({"ID": entry, "Bad": badcount})

    print(f"Overall count processed in {pdb_dir}: {count}")
    print(f"Bad files found and removed: {badcount}")
    print(f"Good files moved to good directory: {count - badcount}")

    return count, badcount


if __name__ == "__main__":
    start_time = time.time()

    check_files(PDB_BASE, GOOD_DIR, BAD_DIR)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_timedelta = timedelta(seconds=elapsed_time)

    print(f"Elapsed time: {elapsed_timedelta} seconds")

# end of file
