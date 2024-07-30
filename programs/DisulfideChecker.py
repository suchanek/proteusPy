#
# Program loads all .ent files in PDB_DIR, checks for SS_BOND
# consistency, and moves good files to the PDB_DIR/good directory,
# moves them to PDB_DIR/bad otherwise.
# Author: Eric G. Suchanek, PhD
# Last modification 7/24/2024


import logging
import os
import shutil
import sys
import time
from datetime import timedelta
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from proteusPy import check_header_from_file, get_logger

_logger = get_logger("DisulfideChecker")

# Paths
HOME_DIR = os.path.expanduser("~")
PDB_BASE = os.getenv("PDB")

if not os.path.isdir(PDB_BASE):
    _logger.error(f"Error: The directory {PDB_BASE} does not exist.")
    sys.exit(1)
else:
    print(f"Found PDB directory at: {PDB_BASE}  ")

GOOD_DIR = os.path.join(PDB_BASE, "good/")
if not os.path.isdir(GOOD_DIR):
    _logger.error(f"Error: The directory {GOOD_DIR} does not exist.")
    sys.exit(1)


BAD_DIR = os.path.join(PDB_BASE, "bad/")
if not os.path.isdir(BAD_DIR):
    _logger.error(f"Error: The directory {BAD_DIR} does not exist.")
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


def check_files(
    pdb_dir: str, good_dir: str, bad_dir: str, verbose=False, quiet=True, cutoff=-1.0
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

    if len(all_pdb_files) == 0:
        print(f"No PDB files! Exiting...")
        return None

    badcount = 0
    bad_entries = []
    count = 0

    pbar = tqdm(all_pdb_files, ncols=80)

    # Using logging_redirect_tqdm to prevent tqdm from interfering with logging
    with logging_redirect_tqdm():
        for fname in pbar:
            entry = name_to_id(fname)

            # Returns > 0 if we can't parse the SSBOND header
            found, errors = check_header_from_file(fname, verbose=verbose)
            if found < 0:
                badcount += 1
                bad_entries.append(entry)
                destination_path = os.path.join(bad_dir, os.path.basename(fname))

                shutil.move(fname, destination_path)
                if verbose:
                    _logger.warning(f"Non-parseable file: {fname} moved to {bad_dir}")
            else:
                sslist = load_disulfides_from_id(
                    entry, verbose=verbose, quiet=quiet, pdb_dir=pdb_dir, cutoff=cutoff
                )

                if sslist is None or len(sslist) == 0:
                    badcount += 1
                    destination_path = os.path.join(bad_dir, os.path.basename(fname))

                    shutil.move(fname, destination_path)
                    if verbose:
                        _logger.warning(f"No SSBonds found: {fname} moved to {bad_dir}")
                else:
                    destination_path = os.path.join(good_dir, os.path.basename(fname))

                    shutil.move(fname, destination_path)
                    if verbose:
                        _logger.info(f"Good file: {fname} moved to {good_dir}")

            count += 1
            pbar.set_postfix({"ID": entry, "Bad": badcount})

    print(f"Overall count processed in {pdb_dir}: {count}")
    print(f"Bad files found and removed: {badcount}")
    print(f"Good files moved to good directory: {count - badcount}")

    return count, badcount


if __name__ == "__main__":
    start_time = time.time()
    count = 0
    badcount = 0
    bad_entries = []

    check_files(PDB_BASE, GOOD_DIR, BAD_DIR, cutoff=8.0, verbose=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_timedelta = timedelta(seconds=elapsed_time)

    print(f"Elapsed time: {elapsed_timedelta} seconds")

# end of file
