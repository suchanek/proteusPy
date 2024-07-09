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


pdblist = PDBList(pdb=GOOD_DIR, verbose=False)
parser = PDBParser(PERMISSIVE=True)

os.chdir(PDB_DIR)
all_pdb_files = glob("*.ent")

print(f"Found: {len(all_pdb_files)} PDB files")
badcount = 0
count = 0

# Loop over all entries,
pbar = tqdm(all_pdb_files, ncols=100)
for entry in pbar:
    # Assuming entry is a filename or a relative path
    entry_path = os.path.abspath(entry)  # Ensure entry_path is an absolute path

    pbar.set_postfix({"Entry": entry, "Bad": badcount})
    if not check_header_from_file(entry):
        badcount += 1
        shutil.copy2(entry, BAD_DIR)
        os.remove(entry)
    else:
        # next attempt to parse the file
        sslist = Extract_Disulfide(entry)
        if sslist is None:
            badcount += 1
            destination_path = os.path.join(BAD_DIR, os.path.basename(entry))

            if not os.path.exists(destination_path):
                shutil.move(entry_path, destination_path)
        else:
            destination_path = os.path.join(GOOD_DIR, os.path.basename(entry))

            if not os.path.exists(destination_path):
                shutil.move(entry_path, destination_path)

    count += 1

print(f"Overall count processed in {PDB_DIR}: {count}")
print(f"Bad files found and removed: {badcount}")
print(f"Good files moved to good directory: {count - badcount}")
