#
# Program loads all .ent files in PDB_DIR, checks for SS_BOND
# consistency, and moves good files to the PDB_DIR/good directory,
# deletes them otherwise.
# Author: Eric G. Suchanek, PhD
# Last modification 11/26/22
#
#

import os
import shutil
from glob import glob

import numpy
from Bio.PDB import PDBList, PDBParser
from Disulfide import check_header_from_file
from tqdm import tqdm

PDB_DIR = "/Users/egs/PDB"
BAD_DIR = PDB_DIR + "/bad"

pdblist = PDBList(pdb=PDB_DIR, verbose=False)
parser = PDBParser(PERMISSIVE=True)

os.chdir(PDB_DIR)
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
        os.remove(entry)
    else:
        if not os.path.exists(f"good/{entry}"):
            shutil.move(entry, "good")
        else:
            os.remove(entry)

    count += 1

print(f"Overall count processed: {count}")
print(f"Bad files found: {badcount}")
