#
# Program reads a comma separated list of PDB IDs and download them
# to the PDB_DIR global. 
# Used to download the list of proteins containing at least one SS bond
# with the ID list generated from: http://www.rcsb.org/
# 
#
# Author: Eric G. Suchanek, PhD
# Last modification 11/26/22
#
# 

import numpy
import os

from tqdm import tqdm
from Bio.PDB import PDBList

PDB_DIR = '/Users/egs/PDB'

def DisulfideLoader(ssfilename='./ss_ids.txt'):
    pdblist = PDBList(pdb=PDB_DIR, verbose=False)

    # list of IDs containing >1 SSBond record
    ssfile = open(ssfilename)
    Line = ssfile.readlines()

    for line in Line:
        entries = line.split(',')

    print(f'Found: {len(entries)} entries')
    completed = {'xxx'} # set to keep track of downloaded

    # file to track already downloaded entries.
    completed_file = open('ss_completed.txt', 'r+')
    donelines = completed_file.readlines()

    for dl in donelines[0]:
        # create a list of pdb id already downloaded
        SS_done = dl.split(',')

    count = len(SS_done) - 1
    completed.update(SS_done) # update the completed set with what's downloaded

    # Loop over all entries, 
    pbar = tqdm(entries, ncols=120)
    for entry in pbar:
        pbar.set_postfix({'Entry': entry})
        if entry not in completed:
            if pdblist.retrieve_pdb_file(entry, file_format='pdb', pdir=PDB_DIR):
                completed.update(entry)
                completed_file.write(f'{entry},')
                count += 1

    completed_file.close()

    print(f'Overall count processed: {count}')

os.chdir(PDB_DIR)
DisulfideLoader()
#
