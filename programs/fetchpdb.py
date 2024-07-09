# This script assumes you have PyMOL installed and are using its API directly.
# PyMOL's API is compatible with Python 3.

import sys

from pymol import cmd, finish_launching

# Initialize PyMOL in command-line mode (no GUI)
finish_launching(["pymol", "-cq"])  # '-cq' for command line quiet mode


def fetch_and_save_pdb(pdbid, save_path):
    """
    Fetches a PDB file using its ID and saves it to the specified path using PyMOL.

    Parameters:
    pdbid (str): The PDB ID to fetch.
    save_path (str): The full path (including filename) where the PDB file will be saved.
    """
    # Fetch the PDB file
    cmd.fetch(pdbid, name=pdbid, type="pdb")

    # Save the PDB file
    cmd.save(save_path, pdbid)

    print(f"File saved as {save_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <pdbid> <save_path>")
        sys.exit(1)

    pdbid = sys.argv[1]
    save_path = sys.argv[2]

    fetch_and_save_pdb(pdbid, save_path)
