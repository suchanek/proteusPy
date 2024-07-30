"""
Program reads a comma separated list of PDB IDs and download them
to the PDB_DIR global.
Used to download the list of proteins containing at least one SS bond
with the ID list generated from: http://www.rcsb.org/

Author: Eric G. Suchanek, PhD
Last modification 7/24/24 -egs-
"""

import logging
import os
import sys
import time
from datetime import timedelta

from Bio.PDB import PDBList
from tqdm import tqdm

import proteusPy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths
HOME_DIR = os.path.expanduser("~")
PDB_DIR = os.getenv("PDB")

if not os.path.isdir(PDB_DIR):
    print(f"Error: The directory {PDB_DIR} does not exist.")
    sys.exit(1)
else:
    print(f"Found PDB directory at: {PDB_DIR}  ")


ids_file_path = os.path.join(PDB_DIR, "ss_ids.txt")
if not os.path.isfile(ids_file_path):
    print(f"Error: The file {ids_file_path} does not exist.")
    sys.exit(1)

input_directory_path = os.path.join(PDB_DIR, "good", "")
if not os.path.isdir(input_directory_path):
    print(f"Error: The directory {input_directory_path} does not exist.")
    sys.exit(1)


def fetch_and_save_pdb(pdbid, save_path=".", verbose=True):
    """
    Fetches a PDB file using its ID and saves it to the specified path using PyMOL.

    Parameters:
    pdbid (str): The PDB ID to fetch.
    save_path (str): The full path where the PDB file will be saved.
    """
    # Ensure the save_path ends with '.pdb'
    from pymol import cmd, finish_launching  # type-ignore

    # Initialize PyMOL in command-line mode (no GUI)
    finish_launching(["pymol", "-cq"])  # '-cq' for command line quiet mode

    filename = f"pdb{pdbid}.ent"
    save_filename = os.path.join(save_path, filename)

    # Fetch the PDB file
    fetched = cmd.fetch(pdbid, name=pdbid, type="pdb")
    if not fetched:
        if verbose:
            logging.error(f"Failed to fetch {pdbid}")
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
        logging.warn("Bad IDs (less than 4 characters):", bad_ids)

    return set(good_ids), good_ids


def get_filenames_from_directory(directory_path):
    """Extracts entry names from filenames in the specified directory."""
    entries = set()
    for filename in os.listdir(directory_path):
        if filename.startswith("pdb") and filename.endswith(".ent"):
            entry_name = filename[3:-4].lower()  # Remove 'pdb' prefix and '.ent' suffix
            entries.add(entry_name)
        else:
            logging.error(
                f"Skipping file: {filename}. Does not comply with naming convention."
            )
    return entries


def find_difference(ids, entries):
    """Finds the difference between two sets."""
    return ids - entries


def DisulfideLoader(idfilename="./ss_ids.txt"):
    pdblist = PDBList(verbose=False)
    formats = ["mmCif", "xml"]
    count = 0
    bad = set()
    entries = set()
    bad_cnt = 0
    fname = ""

    completed = get_filenames_from_directory(input_directory_path)
    entries, _ = read_ids_from_file(ids_file_path)
    difference = find_difference(entries, completed)

    print(f"Already downloaded: {len(completed)} entries")
    print(f"Remaining: {len(difference)} entries")

    # Loop over all entries,
    pbar = tqdm(difference, ncols=80)
    for entry in pbar:
        pbar.set_postfix({"Ent": entry, "Bad": bad_cnt})
        if entry not in completed:
            try:
                # if fetch_and_save_pdb(entry, save_path=PDB_DIR):
                fname = pdblist.retrieve_pdb_file(
                    entry, file_format="pdb", pdir=PDB_DIR
                )
                if fname != "":
                    completed.add(entry)
                    count += 1
                else:
                    found = False
                    for format in formats:
                        fname = pdblist.retrieve_pdb_file(
                            entry, file_format=format, pdir=PDB_DIR
                        )
                        if fname != "":
                            completed.add(entry)
                            count += 1
                            logging.error(
                                f"Downloaded {entry} with {format} format, saved as {fname}"
                            )
                            found = True
                            break
                        else:
                            logging.error(
                                f"Failed to download {entry} with {format} format, retrying... "
                            )
                            continue
                    if not found:
                        logging.error(f"Failed to download {entry}, skipping... ")
                        bad.add(entry)
                        bad_cnt += 1

            except Exception as e:
                logging.error(f"Failed to download {entry}: {e}")
                bad.add(entry)
                bad_cnt += 1

    print(f"Overall files processed: {count}")
    print(f"Non-parsabe files: {bad_cnt}")

    # Write the completed and bad sets to files
    with open(PDB_DIR + "/completed_entries.txt", "w") as comp_file:
        for entry in completed:
            comp_file.write(f"{entry}\n")

    with open(PDB_DIR + "/bad_entries.txt", "w") as bad_file:
        for entry in bad:
            bad_file.write(f"{entry}\n")


start_time = time.time()

os.chdir(PDB_DIR)
DisulfideLoader(idfilename="data/pdb_ids_240709.txt")

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_timedelta = timedelta(seconds=elapsed_time)

print(f"Elapsed time: {elapsed_timedelta} seconds")
