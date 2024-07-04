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

pdblist = PDBList(pdb=GOOD_DIR, verbose=False)
parser = PDBParser(PERMISSIVE=True)

os.chdir(GOOD_DIR)
all_pdb_files = glob("*.ent")

print(f"Found: {len(all_pdb_files)} PDB files")

badcount = 0
count = 0

# Loop over all entries,
pbar = tqdm(all_pdb_files, ncols=100)
for entry in pbar:
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
            shutil.copy2(entry, BAD_DIR)
            os.remove(entry)
        else:
            destination_path = os.path.join(GOOD_DIR, entry)
            if not os.path.exists(destination_path):
                shutil.move(entry, GOOD_DIR)

    count += 1

print(f"Overall count processed in {PDB_DIR}: {count}")
print(f"Bad files found and removed: {badcount}")
print(f"Good files moved to good directory: {count - badcount}")
