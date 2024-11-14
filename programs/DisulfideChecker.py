"""
Program loads all .ent files in PDB_DIR, checks for SS_BOND
consistency, and moves good files to the PDB_DIR/good directory,
moves them to PDB_DIR/bad otherwise.
Author: Eric G. Suchanek, PhD
Last modification 11/11/2024
"""

import os
import shutil
import sys
import time
from glob import glob

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from proteusPy import (
    create_logger,
    extract_ssbonds_and_atoms,
    print_disulfide_bond_info_dict,
)

_logger = create_logger("DisulfideChecker")

# Paths
HOME_DIR = os.path.expanduser("~")
PDB_BASE = os.getenv("PDB")

if not os.path.isdir(PDB_BASE):
    _logger.error("Error: The directory %s does not exist.", PDB_BASE)
    sys.exit(1)
else:
    print(f"Found PDB directory at: {PDB_BASE}  ")

GOOD_DIR = os.path.join(PDB_BASE, "good/")
if not os.path.isdir(GOOD_DIR):
    _logger.error("Error: The directory %s does not exist.", GOOD_DIR)
    sys.exit(1)


BAD_DIR = os.path.join(PDB_BASE, "bad/")
if not os.path.isdir(BAD_DIR):
    _logger.error("Error: The directory %s does not exist.", BAD_DIR)
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


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@timer
def check_files(
    pdb_dir: str,
    good_dir: str,
    bad_dir: str,
    verbose=False,
    quiet=True,
    cutoff=-1.0,
) -> tuple:
    """
    Check all PDB files in the directory `pdb_dir` for SS bond consistency.

    This function processes each PDB file in the specified directory, checking for
    disulfide bond consistency. Files that pass the check are moved to `good_dir`,
    while files that fail are moved to `bad_dir`.

    :param pdb_dir: The directory containing the PDB files to check.
    :type pdb_dir: str
    :param good_dir: The directory to move good files to.
    :type good_dir: str
    :param bad_dir: The directory to move bad files to.
    :type bad_dir: str
    :param verbose: If True, enables verbose logging. Default is False.
    :type verbose: bool
    :param quiet: If True, suppresses non-critical output. Default is True.
    :type quiet: bool

    :return: None
    """

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
        print("No PDB files! Exiting...")
        return None

    badcount = 0
    bad_entries = []
    count = 0

    pbar = tqdm(all_pdb_files, ncols=80)

    # Using logging_redirect_tqdm to prevent tqdm from interfering with logging
    with logging_redirect_tqdm():
        for fname in pbar:
            errors = 0
            found = 0
            entry = name_to_id(fname)
            pbar.set_postfix({"ID": entry, "Bad": badcount})

            # Returns > 0 if we can't parse the SSBOND header

            found, errors = check_file(fname, verbose=verbose)
            if found <= 0:
                badcount += 1
                bad_entries.append(entry)
                destination_path = os.path.join(bad_dir, os.path.basename(fname))

                shutil.move(fname, destination_path)
                if verbose:
                    _logger.warning(
                        "Non-parseable file: %s had %d ssbonds with %d errors, moved to %s",
                        fname,
                        found,
                        errors,
                        bad_dir,
                    )
            else:
                destination_path = os.path.join(good_dir, os.path.basename(fname))

                shutil.move(fname, destination_path)
                if verbose:
                    _logger.info("Good file: %s moved to %s", fname, good_dir)

            count += 1

    print(f"Overall count processed in {pdb_dir}: {count}")
    print(f"Bad files found and removed: {badcount}")
    print(f"Good files moved to good directory: {count - badcount}")

    return count, badcount


def check_file(
    fname: str,
    verbose=False,
    dbg=False,
) -> tuple:
    """
    Check a PDB file in the directory `pdb_dir` for SS bond consistency.

    This function processes a PDB file in the specified directory, checking for
    disulfide bond consistency. Files that pass the check are moved to `good_dir`,
    while files that fail are moved to `bad_dir`.

    :param pdb_dir: The directory containing the PDB files to check.
    :type pdb_dir: str
    :param good_dir: The directory to move good files to.
    :type good_dir: str
    :param bad_dir: The directory to move bad files to.
    :type bad_dir: str
    :param verbose: If True, enables verbose logging. Default is False.
    :type verbose: bool
    :param quiet: If True, suppresses non-critical output. Default is True.
    :type quiet: bool

    :return: None
    """

    if os.path.exists(fname) is False:
        print(f"File {fname} does not exist! Exiting...")
        return None

    # Returns > 0 if we can't parse the SSBOND header
    ssbond_dict, found, errors = extract_ssbonds_and_atoms(
        fname, verbose=verbose, dbg=dbg
    )

    if dbg:
        _logger.info("Found: %d errors: %d", found, errors)
        print_disulfide_bond_info_dict(ssbond_dict)

    return found, errors


if __name__ == "__main__":
    check_files(PDB_BASE, GOOD_DIR, BAD_DIR, cutoff=8.0, verbose=False)


# end of file
