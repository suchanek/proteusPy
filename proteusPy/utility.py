"""
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: BSD\n
Copyright (c)2024 Eric G. Suchanek, PhD, all rights reserved
"""

# Last modification 2025-01-04 12:40:28 -egs-

# pylint: disable=c0103
# pylint: disable=c0301
# pylint: disable=c0302
# pylint: disable=c0413
# pylint: disable=c0412
# pylint: disable=c0415
# pylint: disable=w1514
# pylint: disable=w0718

import copy
import datetime
import glob
import itertools
import logging
import math
import os
import pickle
import platform
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image, ImageFont
from tqdm.contrib.logging import logging_redirect_tqdm

# from proteusPy.DisulfideBase import DisulfideList
from proteusPy.DisulfideExceptions import DisulfideIOException
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    FONTSIZE,
    MODEL_DIR,
    PDB_DIR,
    PROBLEM_ID_FILE,
    SS_ID_FILE,
    SS_PICKLE_FILE,
)

_logger = create_logger(__name__)

# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__  # type: ignore
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


PBAR_COLS = 80


def distance_squared(p1: np.array, p2: np.array) -> np.array:
    """
    Return the square of the N-dimensional distance between the
    two arrays.

    :param np.array p1: N-dimensional array 1
    :param np.array p2: N-dimensional array 2
    :return: np.array N-dimensional distance squared Å^2

    Example
    >>> from proteusPy import distance_squared
    >>> p1 = np.array([1.0, 0.0, 0.0])
    >>> p2 = np.array([0, 1.0, 0])
    >>> d = distance_squared(p1, p2)
    >>> d == float(2)
    np.True_
    """
    return np.sum(np.square(np.subtract(p1, p2)))


def get_jet_colormap(steps):
    """
    Return an array of uniformly spaced RGB values using the 'viridis' colormap.

    :param steps: The number of steps in the output array.

    :return: An array of uniformly spaced RGB values using the 'viridis' colormap. The shape
    of the array is (steps, 3).
    :rtype: numpy.ndarray

    """

    norm = np.linspace(0.0, 1.0, steps)
    colormap = plt.get_cmap("viridis")
    rgbcol = colormap(norm, bytes=True)[:, :3]

    return rgbcol


def grid_dimensions(n: int) -> tuple:
    """
    Compute the number of rows and columns needed to display a list of length `n`.

    Args:
        n (int): Length of input list

    Returns:
        tuple: Number of rows and columns required to display input list
    """
    if n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    else:
        root = math.sqrt(n)
        cols = math.ceil(root)
        rows = cols - 1 if cols * (cols - 1) >= n else cols
        return rows, cols


# given the full dictionary, walk through all the keys (PDB ID)
# for each PDB_ID SS list, find and extract the SS for the first chain
# update the 'pruned' dict with the now shorter SS list


def extract_firstchain_ss(sslist, verbose=False):
    """
    Extract disulfides from the first chain found in
    the SSdict, returns them as a DisulfideList along with the
    number of Xchain disulfides.

    :param sslist: Starting SS list
    :return: (Pruned SS list, xchain)
    """
    from proteusPy.DisulfideBase import DisulfideList

    chain = ""
    chainlist = []
    pc = dc = ""
    res = DisulfideList([], sslist.pdb_id)
    xchain = 0

    # build list of chains
    for ss in sslist:
        pc = ss.proximal_chain
        dc = ss.distal_chain
        if pc != dc:
            xchain += 1
            if verbose:
                _logger.info("extract_firstchain_ss(): Cross chain ss: %s", ss)
        chainlist.append(pc)
    try:
        chain = chainlist[0]
    except IndexError:
        _logger.warning(
            "extract_firstchain_ss(): No chains found in SS list: %s", chain
        )
        return res, xchain

    for ss in sslist:
        if ss.proximal_chain == chain:
            res.append(ss)

    return res, xchain


def prune_extra_ss(sslist):
    """
    Given a dict of disulfides, check for extra chains, grab only the disulfides from
    the first chain and return a dict containing only the first chain disulfides

    :param ssdict: input dictionary with disulfides
    """
    from proteusPy.DisulfideBase import DisulfideList

    xchain = 0

    # print(f'Processing: {ss} with: {sslist}')
    sid = sslist.pdb_id
    pruned_list = DisulfideList([], sid)
    pruned_list, xchain = extract_firstchain_ss(sslist)

    return copy.deepcopy(pruned_list), xchain


def download_file(url: str, directory: str | Path, verbose: bool = False) -> None:
    """
    Download the given URL to the input directory

    :param url: File URL
    :param directory: Directory path for saving the file
    :param verbose: Verbosity, defaults to False
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    elif not os.access(directory, os.W_OK):
        raise OSError(f"Directory not writable: {directory}")

    file_name = url.split("/")[-1]
    file_path = directory / file_name

    if not file_path.exists():
        if verbose:
            _logger.info("Downloading %s...", file_name)
        try:
            command = ["wget", "-P", str(directory), url]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if verbose:
                _logger.info(
                    "Download complete with result %s for: %s", result, file_name
                )
        except subprocess.CalledProcessError as e:
            _logger.error("Download failed: %s\nError: %s", file_name, e.stderr)
            raise
    else:
        if verbose:
            _logger.info("File already exists: %s", file_path)


def get_memory_usage():
    """
    Returns the memory usage of the current process in megabytes (MB).

    Returns:
        float: The memory usage of the current process in megabytes (MB).
    """

    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)


def get_object_size_mb(obj):
    """Return the size of the object in MB."""
    from pympler import asizeof

    size_bytes = asizeof.asizeof(obj)
    size_mb = size_bytes / (1024**2)
    return size_mb


def print_memory_used():
    """
    Print memory used by the proteusPy process (GB).
    """
    from proteusPy import __version__

    mem = get_memory_usage() / (1024**3)  # to GB

    print(f"proteusPy {__version__}: Memory Used: {mem:.2f} GB")


def image_to_ascii_art(fname: str, nwidth: int) -> None:
    """
    Convert an image to ASCII art of given text width.

    Function takes an input filename and width and prints an ASCII art representation to console.

    :param fname: Input filename.
    :param nwidth: Output width in characters.
    """

    if nwidth < 1:
        raise ValueError("Width must be at least 1 character")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Image file not found: {fname}")

    # ASCII character set from darkest to lightest
    CHAR_SET = ["@", "#", "8", "&", "o", ":", "*", ".", " "]

    # Open and resize image
    img = Image.open(fname)
    width, height = img.size
    aspect_ratio = height / width
    new_height = int(
        aspect_ratio * nwidth * 0.55
    )  # 0.55 compensates for character aspect ratio
    img = img.resize((nwidth, new_height))
    img = img.convert("L")  # Convert to grayscale

    # Convert pixel values to ASCII characters using numpy for better performance
    pixels = np.array(img.getdata())
    # Normalize to [0, len(CHAR_SET)-1] range
    pixel_indices = (
        (pixels - pixels.min())
        * (len(CHAR_SET) - 1)
        / (pixels.max() - pixels.min() + 1e-8)
    ).astype(int)
    chars = [CHAR_SET[idx] for idx in pixel_indices]

    # Generate and print ASCII art
    ascii_art = "\n".join(
        "".join(chars[i : i + nwidth]) for i in range(0, len(chars), nwidth)
    )
    print(ascii_art)


def generate_vector_dataframe(base=2):
    """
    Generate a pandas DataFrame containing all combinations for a vector of length 5
    with a given base.

    The function generates combinations of states based on the provided base and
    returns them in a pandas DataFrame. The states are defined as follows:
    - Base 2: ["-", "+"]
    - Base 4: ["1", "2", "3", "4"]
    - Base 6: ["1", "2", "3", "4", "5", "6"]
    - Base 8: ["1", "2", "3", "4", "5", "6", "7", "8"]

    Args:
        base (int): The base for generating the vector states. Supported bases are 2, 4, 6, and 8.

    Returns:
        pandas.DataFrame: A DataFrame containing all combinations of the vector states
        with columns ["chi1", "chi2", "chi3", "chi4", "chi5"].

    Raises:
        ValueError: If the provided base is not supported.
    """
    if base == 2:
        states = ["-", "+"]
    elif base == 4:
        states = ["1", "2", "3", "4"]
    elif base == 6:
        states = ["1", "2", "3", "4", "5", "6"]
    elif base == 8:
        states = ["1", "2", "3", "4", "5", "6", "7", "8"]
    else:
        raise ValueError("Unsupported base")

    combinations = list(itertools.product(states, repeat=5))
    df = pd.DataFrame(combinations, columns=["chi1", "chi2", "chi3", "chi4", "chi5"])
    return df


def sort_by_column(df, column):
    """
    Sort a Pandas DataFrame by the values in the 'incidence' column in descending order.

    :param df: The input DataFrame to be sorted.
    :type df: pandas.DataFrame
    :return: The sorted DataFrame.
    :rtype: pandas.DataFrame
    """
    sorted_df = df.sort_values(by=[column], ascending=False)
    return sorted_df


def retrieve_git_lfs_files(repo_url, objects):
    """
    Retrieves a git-lfs json object from a specified repo.
    It does NOT download the file.
    """
    import requests

    batch_url = f"{repo_url.rstrip('/')}/info/lfs/objects/batch"
    headers = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-type": "application/json",
    }
    data = {"operation": "download", "transfer": ["basic"], "objects": objects}

    response = requests.post(batch_url, headers=headers, json=data, timeout=10)
    if response.status_code == 200:
        # Process the response or save the files
        # For example, you can access the file contents using response.json()
        # and save them to the desired location on your system.
        return response.json()
    else:
        # Handle error case
        print(f"Error: {response.status_code} - {response.text}")
        return None


def display_ss_pymol(
    pdbid: str,
    proximal: int,
    distal: int,
    chain: str,
    solvent: bool = True,
    ray=True,
    width=800,
    height=600,
    dpi=300,
    fname="ss_pymol.png",
) -> None:
    """
    Visualizes specific residues within a given protein structure in PyMOL.

    This function fetches a protein structure by its PDB ID, removes all other chains except
    the specified one, and then shows the proximal and distal residues in spheres. Optionally,
    it can also show solvent molecules as spheres.

    Parameters:
    - pdbid (str): The PDB ID of the protein structure to visualize.
    - proximal (int): The residue number of the proximal residue to visualize.
    - distal (int): The residue number of the distal residue to visualize.
    - chain (str): The chain identifier of the residues to visualize.
    - solvent (bool, optional): Whether to show solvent molecules. Defaults to True.

    Returns:
    - None
    """
    import pymolPy3

    pm = pymolPy3.pymolPy3()

    pm(f"fetch {pdbid}")
    pm(f"remove not chain {chain}")
    pm(f"show spheres, resi {proximal}+{distal}")
    if solvent:
        pm("show spheres, solvent")

    pm(f"color green, resi {proximal}+{distal}")
    pm(f"zoom resi {proximal}+{distal}")
    if ray:
        pm(f"ray {width}, {height}")
        pm(f"png {fname}, dpi={dpi}")

    input("Press Enter to continue...")
    return None


def Download_Disulfides(
    pdb_home=PDB_DIR, model_home=MODEL_DIR, verbose=False, reset=False
) -> None:
    """
    Read a comma separated list of PDB IDs and download them to the ``pdb_home`` path.

    This utility function is used to download proteins containing at least one
    SS bond with the ID list generated from: http://www.rcsb.org/.

    This is the primary data loader for the proteusPy Disulfide analysis package.
    The list of IDs represents files in the RCSB containing > 1 disulfide bond,
    and it contains over 39000 structures. The total download takes about 12 hours.
    The function keeps track of downloaded files so it's possible to interrupt and
    restart the download without duplicating effort.

    :param pdb_home: Path for downloaded files, defaults to PDB_DIR
    :param model_home: Path for extracted data, defaults to MODEL_DIR
    :param verbose: Verbosity, defaults to False
    :param reset: Reset the downloaded files index. Used to restart the download.
    :raises DisulfideIOException: I/O error raised when the PDB file is not found.
    """
    from Bio.PDB import PDBList

    start = time.time()
    donelines = []
    SS_done = []
    ssfile = None

    cwd = os.getcwd()
    os.chdir(pdb_home)

    pdblist = PDBList(verbose=verbose)
    ssfilename = f"{model_home}{SS_ID_FILE}"
    print(ssfilename)

    # list of IDs containing >1 SSBond record
    try:
        ssfile = open(ssfilename, "r")
        Line = ssfile.readlines()
    except Exception:
        raise DisulfideIOException(f"Cannot open file: {ssfile}")

    for line in Line:
        entries = line.split(",")

    print(f"Found: {len(entries)} entries")
    completed = {"xxx"}  # set to keep track of downloaded

    # file to track already downloaded entries.
    if reset is True:
        completed_file = open(f"{model_home}ss_completed.txt", "w")
        donelines = []
    else:
        completed_file = open(f"{model_home}ss_completed.txt", "w+")
        donelines = completed_file.readlines()

    if len(donelines) > 0:
        for dl in donelines[0]:
            # create a list of pdb id already downloaded
            SS_done = dl.split(",")

    count = len(SS_done) - 1
    completed.update(SS_done)  # update the completed set with what's downloaded

    # Loop over all entries,
    pbar = tqdm(entries, ncols=PBAR_COLS)
    for entry in pbar:
        pbar.set_postfix({"Entry": entry})
        if entry not in completed:
            if pdblist.retrieve_pdb_file(entry, file_format="pdb", pdir=pdb_home):
                completed.update(entry)
                completed_file.write(f"{entry},")
                count += 1

    completed_file.close()

    end = time.time()
    elapsed = end - start

    print(f"Overall files processed: {count}")
    print(f"Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)")
    os.chdir(cwd)
    return


def remove_duplicate_ss(sslist) -> list:
    """Remove duplicate disulfides from the input list."""
    pruned = []
    for ss in sslist:
        if ss not in pruned:
            pruned.append(ss)
    return pruned


# Function extracts the disulfide bonds from the PDB files and creates the .pkl files
# needed for the proteusPy.DisulfideLoader class. This function is largely
# replaced by the DisulfideExtractor_mp.py program which uses multiprocessing to
# speed up the extraction process. The function is retained for reference.


def Extract_Disulfides(
    numb=-1,
    verbose=False,
    quiet=True,
    pdbdir=PDB_DIR,
    baddir=Path(PDB_DIR) / "bad",
    datadir=MODEL_DIR,
    picklefile=SS_PICKLE_FILE,
    problemfile=PROBLEM_ID_FILE,
    dist_cutoff=-1.0,
    prune=True,
) -> None:
    """
    Read the PDB files contained in ``pdbdir`` and create the .pkl files needed for the
    proteusPy.DisulfideLoader.DisulfideLoader class.
    The ```Disulfide``` objects are contained in a ```DisulfideList``` object and
    ```Dict``` within these files. In addition, .csv files containing all of
    the torsions for the disulfides and problem IDs are written. The optional
    ```dist_cutoff``` allows for removal of Disufides whose Cα-Cα distance is >
    than the cutoff value. If it's -1.0 then the function keeps all Disulfides.

    :param numb:           Number of entries to process, defaults to all
    :param verbose:        More messages
    :param quiet:          Turn off DisulfideConstruction warnings
    :param pdbdir:         Path to PDB files
    :param datadir:        Path to resulting .pkl files
    :param picklefile:     Name of the disulfide .pkl file
    :param problemfile:    Name of the .csv file containing problem ids
    :param dist_cutoff:    Ca distance cutoff to reject a Disulfide.
    :param prune:          Move bad files to bad directory, defaults to True
    """
    from proteusPy.DisulfideBase import DisulfideList
    from proteusPy.DisulfideIO import load_disulfides_from_id

    def name_to_id(fname: str) -> str:
        """
        Returns the PDB ID from the filename.

        :param fname: Complete PDB filename
        :return: PDB ID
        """
        ent = fname[3:-4]
        return ent

    if quiet:
        _logger.setLevel(logging.ERROR)

    bad_dir = baddir

    entrylist = []
    sslist = []
    problem_ids = []
    bad = bad_dist = tot = cnt = i = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_list = DisulfideList([], "PDB_SS")
    All_ss_dict2 = {}  # new dict of pointers to indices

    start = time.time()
    cwd = os.getcwd()

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    os.chdir(pdbdir)

    ss_filelist = glob.glob("*.ent")

    # the filenames are in the form pdb{entry}.ent, I loop through them and extract
    # the PDB ID, with Disulfide.name_to_id(), then add to entrylist.

    for entry in ss_filelist:
        entrylist.append(name_to_id(entry))

    if verbose:
        _logger.info("PDB Ids: %s, len: %s", entrylist, len(entrylist))

    if numb < 0:
        tot = len(entrylist)
    else:
        tot = numb

    # create a dataframe with the following columns for the disulfide conformations
    # extracted from the structure

    # SS_df = pd.DataFrame(columns=Torsion_DF_Cols)

    # define a tqdm progressbar using the fully loaded entrylist list.
    # If numb is passed then
    # only do the last numb entries.

    # loop over ss_filelist, create disulfides and initialize them
    # the logging_redirect_tqdm() context manager will redirect the logging output
    # to the tqdm progress bar.

    with logging_redirect_tqdm(loggers=[_logger]):
        if numb > 0:
            pbar = tqdm(entrylist[:numb], ncols=PBAR_COLS, mininterval=0.5)
        else:
            pbar = tqdm(entrylist, ncols=PBAR_COLS, mininterval=0.5)

        for entry in pbar:
            _sslist = DisulfideList([], entry)

            # returns an empty list if none are found.
            _sslist = load_disulfides_from_id(
                entry,
                verbose=verbose,
                quiet=quiet,
                pdb_dir=pdbdir,
                cutoff=dist_cutoff,
            )

            # sslist, xchain = prune_extra_ss(_sslist)
            # sslist = _sslist

            if len(_sslist) > 0:
                sslist = remove_duplicate_ss(_sslist)
                sslist2 = []  # list to hold indices for ss_dict2
                for ss in sslist:
                    All_ss_list.append(ss)
                    sslist2.append(cnt)
                    cnt += 1
                    tot += 1

                # All_ss_dict[entry] = sslist
                # print(f'Entry: {entry}. Dict indices: {sslist2}')
                All_ss_dict2[entry] = sslist2

                # print(f'{entry} ss dict adding: {sslist2}')

            else:  ## _sslist is empty!
                bad += 1
                problem_ids.append(entry)
                if verbose:
                    _logger.warning(
                        "Extract_Disulfides(): No SS parsed for: %s!", entry
                    )

                if prune:
                    fname = f"pdb{entry}.ent"
                    # Construct the full path for the new destination file
                    destination_file_path = Path(bad_dir) / fname

                    # Copy the file to the new destination with the correct filename
                    _logger.warning(
                        "Extract_Disulfides(): Moving %s to %s",
                        fname,
                        destination_file_path,
                    )
                    shutil.move(fname, destination_file_path)
                continue  ## this entry has no SS bonds, so we break the loop

            i += 1
            if i % 100 == 0:
                pbar.set_postfix({"ID": entry, "NoSS": bad, "Cnt": cnt})

    pbar.close()

    if bad > 0:
        prob_cols = ["id"]
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df["id"] = problem_ids

        _logger.warning(
            "Found and moved: %d non-parsable structures. Saving problem IDs to file: %s",
            len(problem_ids),
            Path(datadir) / problemfile,
        )

        problem_df.to_csv(Path(datadir) / problemfile)
    else:  ## no bad files found
        if verbose:
            _logger.info("No non-parsable structures found.")

    if bad_dist > 0:
        if verbose:
            _logger.warning("Found and ignored: %s long SS bonds.", bad_dist)

    else:
        if verbose:
            _logger.info("Extract_Disulfides(): No problems found.")

    # dump the all_ss list of disulfides to a .pkl file. ~520 MB.
    fname = Path(datadir) / picklefile

    if verbose:
        _logger.info("Saving %d Disulfides to file: %s", len(All_ss_list), fname)

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_list, f)

    end = time.time()
    elapsed = end - start

    if verbose:
        _logger.info(
            "Disulfide Extraction complete! Elapsed time: %s (h:m:s)",
            datetime.timedelta(seconds=elapsed),
        )

    # return to original directory
    os.chdir(cwd)

    # restore the logger level
    if quiet:
        _logger.setLevel(logging.WARNING)
    return


def Extract_Disulfides_From_List(
    numb=-1,
    verbose=False,
    quiet=True,
    pdbdir=PDB_DIR,
    baddir=PDB_DIR + "/bad/",
    datadir=MODEL_DIR,
    picklefile=SS_PICKLE_FILE,
    problemfile=PROBLEM_ID_FILE,
    dist_cutoff=-1.0,
    prune=True,
    sslist=None,
) -> None:
    """
    Read the PDB files contained in ``pdbdir`` and create the .pkl files needed for the
    proteusPy.DisulfideLoader.DisulfideLoader class.
    The ```Disulfide``` objects are contained in a ```DisulfideList``` object and
    ```Dict``` within these files. In addition, .csv files containing all of
    the torsions for the disulfides and problem IDs are written. The optional
    ```dist_cutoff``` allows for removal of Disufides whose Cα-Cα distance is >
    than the cutoff value. If it's -1.0 then the function keeps all Disulfides.

    :param numb:           Number of entries to process, defaults to all
    :param verbose:        More messages
    :param quiet:          Turn off DisulfideConstruction warnings
    :param pdbdir:         Path to PDB files
    :param datadir:        Path to resulting .pkl files
    :param picklefile:     Name of the disulfide .pkl file
    :param torsionfile:    Name of the disulfide torsion file .csv created
    :param problemfile:    Name of the .csv file containing problem ids
    :param dist_cutoff:    Ca distance cutoff to reject a Disulfide.
    :param prune:          Move bad files to bad directory, defaults to True
    """
    from proteusPy.DisulfideBase import DisulfideList
    from proteusPy.DisulfideIO import load_disulfides_from_id

    if quiet:
        _logger.setLevel(logging.ERROR)

    bad_dir = baddir

    entrylist = []
    problem_ids = []
    bad = bad_dist = tot = cnt = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_list = DisulfideList([], "PDB_SS")
    All_ss_dict2 = {}  # new dict of pointers to indices

    start = time.time()
    cwd = os.getcwd()

    os.chdir(pdbdir)

    entrylist = sslist
    if verbose:
        _logger.info(
            "Extract_Disulfides(): PDB Ids: %s, len: %d",
            entrylist,
            len(entrylist),
        )

    with logging_redirect_tqdm(loggers=[_logger]):
        if numb > 0:
            pbar = tqdm(entrylist[:numb], ncols=PBAR_COLS)
        else:
            pbar = tqdm(entrylist, ncols=PBAR_COLS)

        for entry in pbar:
            _sslist = DisulfideList([], entry)

            # returns an empty list if none are found.
            _sslist = load_disulfides_from_id(
                entry,
                verbose=verbose,
                quiet=quiet,
                pdb_dir=pdbdir,
                cutoff=dist_cutoff,
            )

            # sslist, xchain = prune_extra_ss(_sslist)
            # sslist = _sslist

            if len(_sslist) > 0:
                sslist = remove_duplicate_ss(_sslist)
                sslist2 = []  # list to hold indices for ss_dict2
                for ss in sslist:
                    All_ss_list.append(ss)
                    sslist2.append(cnt)
                    cnt += 1
                    tot += 1

                # All_ss_dict[entry] = sslist
                # print(f'Entry: {entry}. Dict indices: {sslist2}')
                All_ss_dict2[entry] = sslist2

                # print(f'{entry} ss dict adding: {sslist2}')

            else:  ## _sslist is empty!
                bad += 1
                problem_ids.append(entry)
                if verbose:
                    _logger.warning(
                        "Extract_Disulfides(): No SS parsed for: %s!", entry
                    )
                if prune:
                    fname = f"pdb{entry}.ent"
                    # Construct the full path for the new destination file
                    destination_file_path = Path(bad_dir) / fname
                    # Copy the file to the new destination with the correct filename
                    _logger.warning(
                        "Extract_Disulfides(): Moving %s to %s",
                        fname,
                        destination_file_path,
                    )
                    shutil.move(fname, destination_file_path)
                continue  ## this entry has no SS bonds, so we break the loop

            pbar.set_postfix(
                {"ID": entry, "NoSS": bad, "Cnt": tot}
            )  # update the progress bar

    if bad > 0:
        prob_cols = ["id"]
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df["id"] = problem_ids

        _logger.warning(
            (
                "Found and moved: %d non-parsable structures.\nSaving problem IDs to file: %s%s",
                len(problem_ids),
                datadir,
                problemfile,
            )
        )

        problem_df.to_csv(Path(datadir / problemfile))

    else:  ## no bad files found
        if verbose:
            _logger.info("Extract_Disulfides(): No non-parsable structures found.")

    if bad_dist > 0:
        if verbose:
            _logger.warning("Found and ignored: %s long SS bonds.", bad_dist)

    else:
        if verbose:
            _logger.info("No problems found.")

    # dump the all_ss list of disulfides to a .pkl file. ~520 MB.
    fname = Path(datadir) / picklefile

    if verbose:
        _logger.info("Saving %d Disulfides to file: %s", len(All_ss_list), fname)

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_list, f)

    end = time.time()
    elapsed = end - start

    if verbose:
        _logger.info(
            "Disulfide Extraction complete! Elapsed time: %s (h:m:s)",
            datetime.timedelta(seconds=elapsed),
        )

    # return to original directory
    os.chdir(cwd)

    # restore the logger level
    if quiet:
        _logger.setLevel(logging.WARNING)
    return


def get_theme() -> str:
    """
    Determine the display theme for the current operating system.

    Returns:
    :return str: 'light' if the theme is light, 'dark' if the theme is dark, and 'light' otherwise
    """
    system = platform.system()

    def _get_macos_theme() -> str:
        script = """
        tell application "System Events"
            tell appearance preferences
                if dark mode is true then
                    return "dark"
                else
                    return "light"
                end if
            end tell
        end tell
        """
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if result.returncode == 0:
                theme = result.stdout.strip().lower()
                if theme in ["dark", "light"]:
                    return theme
        except subprocess.CalledProcessError as e:
            _logger.error("Failed to get macOS theme: %s", e.stderr)
        except Exception as e:
            _logger.error("Error getting macOS theme: %s", e)
        return "light"

    def _get_windows_theme() -> str:
        try:
            # Lazy import winreg only on Windows
            from winreg import (
                HKEY_CURRENT_USER,
                CloseKey,
                ConnectRegistry,
                OpenKey,
                QueryValueEx,
            )

            registry = ConnectRegistry(None, HKEY_CURRENT_USER)
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            key = OpenKey(registry, key_path)
            try:
                value, _ = QueryValueEx(key, "AppsUseLightTheme")
                return "dark" if value == 0 else "light"
            finally:
                CloseKey(key)
        except ImportError:
            _logger.warning("winreg module not available")
        except Exception as e:
            _logger.error("Failed to get Windows theme: %s", e)
        return "light"

    def _get_linux_theme() -> str:
        try:
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if result.returncode == 0 and "dark" in result.stdout.strip().lower():
                return "dark"
        except Exception as e:
            _logger.error("Failed to get Linux theme: %s", e)
        return "light"

    # Use a dictionary to map system to theme getter function
    theme_getters = {
        "Darwin": _get_macos_theme,
        "Windows": _get_windows_theme,
        "Linux": _get_linux_theme,
    }

    # Get theme using appropriate function or default to light
    return theme_getters.get(system, lambda: "light")()


def save_list_to_file(input_list, filename):
    """
    Save the input list to a file using pickle.

    :param input_list: List to be saved.
    :type input_list: list
    :param filename: Name of the file where the list will be saved.
    :type filename: str
    """
    with open(filename, "wb") as file:
        pickle.dump(input_list, file)


def load_list_from_file(filename):
    """
    Load a list from a file using pickle.

    This function reads a list from a file that was previously saved using the pickle module.

    :param filename: Name of the file from which the list will be loaded.
    :type filename: str
    :return: The list that was loaded from the file.
    :rtype: list
    """
    with open(filename, "rb") as file:
        loaded_list = pickle.load(file)
    return loaded_list


def set_plotly_theme(theme: str, verbose=False) -> str:
    """
    Set the Plotly theme based on the provided theme parameter.

    This function sets the default Plotly template to either 'plotly_white' or 'plotly_dark'
    based on the input theme. If 'auto' is selected, the theme is determined automatically
    based on the current system or application theme.

    :param theme: The theme to set for Plotly. Must be 'auto', 'light', or 'dark'.
    :type theme: str
    :param verbose: If True, logs the selected theme. Defaults to False.
    :type verbose: bool, optional
    :raises ValueError: If an invalid theme is provided.
    :return: The current Plotly template.
    :rtype: str
    """
    from plotly import io as pio

    match theme.lower():
        case "auto":
            _theme = get_theme()
            if _theme == "light":
                pio.templates.default = "plotly_white"
            else:
                pio.templates.default = "plotly_dark"
        case "light":
            pio.templates.default = "plotly_white"
        case "dark":
            pio.templates.default = "plotly_dark"
        case _:
            raise ValueError("Invalid theme. Must be 'auto', 'light', or 'dark'.")

    if verbose:
        _logger.info("Plotly theme set to: %s", pio.templates.default)

    return pio.templates.default


def set_pyvista_theme(theme: str, verbose=False) -> str:
    """
    Set the PyVista theme based on the provided theme parameter.

    This function sets the default PyVista theme to either 'document' or 'dark'
    based on the input theme. If 'auto' is selected, the theme is determined automatically
    based on the current system or application theme.

    :param theme: The theme to set for PyVista. Must be 'auto', 'light', or 'dark'.
    :type theme: str
    :param verbose: If True, logs the selected theme. Defaults to False.
    :type verbose: bool, optional
    :raises ValueError: If an invalid theme is provided.
    :return: The current PyVista theme.
    :rtype: str
    """
    import pyvista as pv

    _theme = get_theme()

    match theme.lower():
        case "auto":
            if _theme == "light":
                pv.set_plot_theme("document")
            else:
                pv.set_plot_theme("dark")
                _theme = "dark"
        case "light":
            pv.set_plot_theme("document")
        case "dark":
            pv.set_plot_theme("dark")
            _theme = "dark"
        case _:
            raise ValueError("Invalid theme. Must be 'auto', 'light', or 'dark'.")

    if verbose:
        _logger.info("PyVista theme set to: %s", _theme.lower())

    return _theme


def find_arial_font():
    """
    Find the system font file arial.ttf for macOS, Windows, and Linux.

    :return: The path to the arial.ttf font file if found, otherwise None.
    """
    font_paths = {
        "Windows": [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\Arial.ttf"],
        "Darwin": [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ],
        "Linux": [
            "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
            "/usr/share/fonts/truetype/msttcore/arial.ttf",
            "/usr/share/fonts/truetype/arial.ttf",
            "/usr/share/fonts/Arial.ttf",
        ],
    }

    system = platform.system()
    if system in font_paths:
        for path in font_paths[system]:
            if os.path.exists(path):
                return path

    return None


def calculate_fontsize(title, window_width, max_fontsize=FONTSIZE, min_fontsize=2):
    """
    Calculate the maximum font size for the title so that it fits within the window width in PyVista.

    :param title: The title text.
    :param window_width: The width of the window in pixels.
    :param font_path: The path to the font file.
    :param max_fontsize: The maximum font size.
    :param min_fontsize: The minimum font size.
    :return: The calculated font size.
    """

    if not title:
        return min_fontsize  # Default to the smallest size if the title is empty

    font_path = find_arial_font()

    if not font_path:
        _logger.warning("Arial font not found.")
        return min_fontsize

    def get_text_width(title, fontsize):
        # Load the font with the given fontsize
        font = ImageFont.truetype(font_path, fontsize)
        # Calculate and return the text width using getbbox
        sz = font.getbbox(title)
        text_width = font.getbbox(title)[2]

        _logger.debug(
            "Font size: %d, bbox: %s, text width: %d", fontsize, sz, text_width
        )

        return text_width

    fontsize = max_fontsize
    while fontsize > min_fontsize:
        text_width = get_text_width(title, fontsize)
        if text_width <= window_width:
            break
        fontsize -= 1

    _logger.debug("Calculated fontsize: %d", fontsize)
    return fontsize // 2


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
