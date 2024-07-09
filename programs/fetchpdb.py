import os
import sys

from pymol import cmd, finish_launching

# Initialize PyMOL in command-line mode (no GUI)
finish_launching(["pymol", "-cq"])  # '-cq' for command line quiet mode


def fetch_and_save_pdb(pdbid, save_path=".", verbose=True):
    """
    Fetches a PDB file using its ID and saves it to the specified path using PyMOL.

    Parameters:
    pdbid (str): The PDB ID to fetch.
    save_path (str): The full path where the PDB file will be saved.
    """
    # Ensure the save_path ends with '.pdb'
    save_filename = os.path.join(save_path, pdbid) + ".pdb"

    # Fetch the PDB file
    cmd.fetch(pdbid, name=pdbid, type="pdb")

    # Save the PDB file
    cmd.save(save_filename, pdbid)
    if verbose:
        print(f"File saved as {save_filename}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <pdbid> <save_path>")
        sys.exit(1)

    pdbid = sys.argv[1]
    save_path = sys.argv[2]

    fetch_and_save_pdb(pdbid, save_path)
