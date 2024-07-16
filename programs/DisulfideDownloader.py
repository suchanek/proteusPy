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

import logging
import os

import numpy
from Bio.PDB import PDBList
from tqdm import tqdm



def fetch_and_save_pdb(pdbid, save_path=".", verbose=True):
    """
    Fetches a PDB file using its ID and saves it to the specified path using PyMOL.

    Parameters:
    pdbid (str): The PDB ID to fetch.
    save_path (str): The full path where the PDB file will be saved.
    """
    # Ensure the save_path ends with '.pdb'
    from pymol import cmd, finish_launching
    
    # Initialize PyMOL in command-line mode (no GUI)
    finish_launching(["pymol", "-cq"])  # '-cq' for command line quiet mode


    filename = f"pdb{pdbid}.ent"
    save_filename = os.path.join(save_path, filename)

    # Fetch the PDB file
    fetched = cmd.fetch(pdbid, name=pdbid, type="pdb")
    if not fetched:
        if verbose:
            print(f"Failed to fetch {pdbid}")
        return False

    # Save the PDB file
    cmd.save(save_filename, pdbid)
    os.remove(f"{pdbid}.pdb")

    if verbose:
        print(f"File saved as {save_filename}")
    return True


def read_ids_from_file(file_path):
    """Reads IDs from a file, prints bad IDs, and returns the good ones as a set."""
    with open(file_path, "r") as file:
        ids = file.read().strip().lower().split(",")

    # Separate good and bad IDs based on length
    good_ids = []
    bad_ids = []
    for id in ids:
        if len(id) == 4:
            good_ids.append(id)
        else:
            bad_ids.append(id)

    # Print the list of bad IDs, if any
    if bad_ids:
        print("Bad IDs (less than 4 characters):", bad_ids)

    return set(good_ids), good_ids


def get_filenames_from_directory(directory_path):
    """Extracts entry names from filenames in the specified directory."""
    entries = set()
    for filename in os.listdir(directory_path):
        if filename.startswith("pdb") and filename.endswith(".ent"):
            entry_name = filename[3:-4].lower()  # Remove 'pdb' prefix and '.ent' suffix
            entries.add(entry_name)
    return entries


def find_difference(ids, entries):
    """Finds the difference between two sets."""
    return ids - entries


# Paths
PDB_DIR = "/Users/egs/PDB"

ids_file_path = os.path.join(PDB_DIR, "ss_ids.txt")
input_directory_path = os.path.join(PDB_DIR, "good", "")
# input_directory_path = PDB_DIR + "/good/"


def DisulfideLoader(idfilename="./ss_ids.txt"):
    pdblist = PDBList(verbose=False)
    count = 0
    bad = set()
    entries = set()
    bad_cnt = 0

    completed = get_filenames_from_directory(input_directory_path)
    entries, _ = read_ids_from_file(ids_file_path)
    difference = find_difference(entries, completed)

    print(f"Already downloaded: {len(completed)} entries")
    print(f"Remaining: {len(difference)} entries")

    # Loop over all entries,
    pbar = tqdm(difference, ncols=120)
    for entry in pbar:
        pbar.set_postfix({"Ent": entry, "Bad": bad_cnt})
        if entry not in completed:
            try:
                #if fetch_and_save_pdb(entry, save_path=PDB_DIR):
                if pdblist.retrieve_pdb_file(entry, file_format="pdb", pdir=PDB_DIR):
                    completed.add(entry)
                    count += 1
                else:
                    bad.add(entry)
                    bad_cnt += 1
            except Exception as e:
                logging.error(f"Failed to download {entry}: {e}")
                bad.add(entry)
                bad_cnt += 1

    print(f"Overall count processed: {count}")
    print(f"Bad entries: {bad_cnt}")

    # Write the completed and bad sets to files
    with open(PDB_DIR + "/completed_entries.txt", "w") as comp_file:
        for entry in completed:
            comp_file.write(f"{entry}\n")

    with open(PDB_DIR + "/bad_entries.txt", "w") as bad_file:
        for entry in bad:
            bad_file.write(f"{entry}\n")


os.chdir(PDB_DIR)
DisulfideLoader(idfilename="data/pdb_ids_240709.txt")
