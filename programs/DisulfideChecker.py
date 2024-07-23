#
# Program loads all .ent files in PDB_DIR, checks for SS_BOND
# consistency, and moves good files to the PDB_DIR/good directory,
# moves them to PDB_DIR/bad otherwise.
# Author: Eric G. Suchanek, PhD
# Last modification 7/4/2024


import os
import shutil
from glob import glob

import numpy
from Bio.PDB import PDBList, PDBParser
from tqdm import tqdm

from proteusPy import Extract_Disulfide, check_header_from_file

PDB_DIR = "/Users/egs/PDB"
GOOD_DIR = PDB_DIR + "/good"
BAD_DIR = PDB_DIR + "/bad"


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


pdblist = PDBList(verbose=False)
parser = PDBParser(PERMISSIVE=True)


def check_files(
    pdb_dir: str, good_dir: str, bad_dir: str, verbose=True, quiet=True
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
    # print(f"{all_pdb_files}")
    if len(all_pdb_files) == 0:
        print(f"No PDB files! Exiting...")
        return None

    badcount = 0
    count = 0

    pbar = tqdm(all_pdb_files, ncols=100)
    for fname in pbar:
        entry = name_to_id(fname)

        if not check_header_from_file(fname, verbose=verbose):
            badcount += 1
            destination_path = os.path.join(bad_dir, os.path.basename(fname))

            if not os.path.exists(destination_path):
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

                if not os.path.exists(destination_path):
                    shutil.move(fname, destination_path)
                print(f"Bad file: {fname} moved to {bad_dir}")

            else:
                destination_path = os.path.join(good_dir, os.path.basename(fname))

                if not os.path.exists(destination_path):
                    shutil.move(fname, destination_path)
                print(f"Good file: {fname} moved to {good_dir}")

        count += 1
        pbar.set_postfix({"ID": entry, "Bad": badcount})

    print(f"Overall count processed in {pdb_dir}: {count}")
    print(f"Bad files found and removed: {badcount}")
    print(f"Good files moved to good directory: {count - badcount}")

    return count, badcount


if __name__ == "__main__":
    check_files(PDB_DIR, GOOD_DIR, BAD_DIR)
