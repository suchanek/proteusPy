"""
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: BSD\n
Copyright (c)2024 Eric G. Suchanek, PhD, all rights reserved
"""

# Last modification 2025-01-04 12:40:28 -egs-

# pylint: disable=c0103
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from tqdm.contrib.logging import logging_redirect_tqdm

from proteusPy import Disulfide, DisulfideList, __version__
from proteusPy.DisulfideExceptions import DisulfideIOException
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
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

    Example:
        >>> get_viridis_colormap(5)
        array([[ 68,   1,  84],
               [ 72,  40, 120],
               [ 32, 144, 140],
               [ 94, 201,  98],
               [253, 231,  37]], dtype=uint8)
    """

    norm = np.linspace(0.0, 1.0, steps)
    colormap = plt.get_cmap("viridis")
    rgbcol = colormap(norm, bytes=True)[:, :3]

    return rgbcol


def Oget_jet_colormap(steps):
    """
    Return an array of uniformly spaced RGB values using the 'jet' colormap.

    :param steps: The number of steps in the output array.

    :return: An array of uniformly spaced RGB values using the 'jet' colormap. The shape
    of the array is (steps, 3).
    :rtype: numpy.ndarray

    Example:
        >>> get_jet_colormap(5)
        array([[  0,   0, 127],
               [  0, 128, 255],
               [124, 255, 121],
               [255, 148,   0],
               [127,   0,   0]], dtype=uint8)
    """

    norm = np.linspace(0.0, 1.0, steps)
    colormap = matplotlib.colormaps["jet"]
    #    colormap = cm.get_cmap("jet", steps)
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

    chain = ""
    chainlist = []
    pc = dc = ""
    res = DisulfideList([], sslist.id)
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
    xchain = 0

    # print(f'Processing: {ss} with: {sslist}')
    sid = sslist.pdb_id
    pruned_list = DisulfideList([], sid)
    pruned_list, xchain = extract_firstchain_ss(sslist)

    return copy.deepcopy(pruned_list), xchain


def download_file(url, directory, verbose=False):
    """
    Download the given URL to the input directory

    :param url: File URL
    :param directory: Directory path for saving the file
    :param verbose: Verbosity, defaults to False
    """
    file_name = url.split("/")[-1]
    file_path = Path(directory) / file_name

    if not os.path.exists(file_path):
        if verbose:
            _logger.info("Downloading %s...", file_name)
        command = ["wget", "-P", directory, url]
        subprocess.run(command, check=True)
        print("Download complete.")
    else:
        print(f"{file_name} already exists in {directory}.")


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
    mem = get_memory_usage() / (1024**3)  # to GB

    print(f"proteusPy {__version__}: Memory Used: {mem:.2f} GB")


def image_to_ascii_art(fname, nwidth):
    """
    Convert an image to ASCII art of given text width.

    Function takes an input filename and width and prints an ASCII art representation to console.

    :param fname: Input filename.
    :param nwidth: Output width in characters.
    """
    from PIL import Image
    from sklearn.preprocessing import minmax_scale  # type: ignore

    # Open the image file
    image = Image.open(fname)

    # Resize the image to reduce the number of pixels
    width, height = image.size
    aspect_ratio = height / width
    new_width = nwidth
    new_height = aspect_ratio * new_width * 0.55  # 0.55 is an adjustment factor
    image = image.resize((new_width, int(new_height)))

    # Convert the image to grayscale.
    image = image.convert("L")

    # Define the ASCII character set to use (inverted colormap)
    char_set = ["@", "#", "8", "&", "o", ":", "*", ".", " "]

    # Normalize the pixel values in the image.
    pixel_data = list(image.getdata())
    pixel_data_norm = minmax_scale(
        pixel_data, feature_range=(0, len(char_set) - 1), copy=True
    )
    pixel_data_norm = [int(x) for x in pixel_data_norm]

    char_array = [char_set[pixel] for pixel in pixel_data_norm]
    # Generate the ASCII art string
    ascii_art = "\n".join(
        [
            "".join([char_array[i + j] for j in range(new_width)])
            for i in range(0, len(char_array), new_width)
        ]
    )

    # Print the ASCII art string.
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


def remove_duplicate_ss(sslist: DisulfideList) -> list[Disulfide]:
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
    from proteusPy.DisulfideList import load_disulfides_from_id

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
    from proteusPy.DisulfideList import load_disulfides_from_id

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


def get_theme():
    """
    Determine the display theme for the current operating system.

    Returns:
    :return str: 'light' if the theme is light, 'dark' if the theme is dark, and 'light' otherwise

    Example:
    >>> get_theme()
    'dark'
    """
    system = platform.system()

    if system == "Darwin":
        # macOS
        try:
            # AppleScript to get the appearance setting
            script = """
            tell application "System Events"
                tell appearance preferences
                    if (dark mode) then
                        return "dark"
                    else
                        return "light"
                    end if
                end tell
            end tell
            """
            # Run the AppleScript
            result = subprocess.run(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Check the output
            if result.returncode == 0:
                theme = result.stdout.strip().lower()
                if theme in ["dark", "light"]:
                    return theme
                return None
            return None

        except Exception:
            # In case of any exception, return None
            return None

    elif system == "Windows":
        # Windows
        try:
            import winreg

            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            key = winreg.OpenKey(registry, key_path)
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)

            if value == 0:
                return "dark"
            else:
                return "light"

        except Exception:
            # In case of any exception, return None
            return None

    elif system == "Linux":
        # Linux
        try:
            # Check for GTK theme setting
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            if result.returncode == 0:
                theme = result.stdout.strip().lower()
                if "dark" in theme:
                    return "dark"
                else:
                    return "light"
            return "light"

        except Exception:
            # In case of any exception, return None
            return "light"

    else:
        # Unsupported OS
        return "light"


# functions to calculate statistics and filter disulfide lists via pandas


def calculate_std_cutoff(df, column, num_std=2):
    """
    Calculate cutoff based on standard deviation.

    :param df: DataFrame containing the deviations.
    :type df: pd.DataFrame
    :param column: Column name for which to calculate the cutoff.
    :type column: str
    :param num_std: Number of standard deviations to use for the cutoff.
    :type num_std: int
    :return: Cutoff value.
    :rtype: float
    """
    mean = df[column].mean()
    std = df[column].std()
    cutoff = mean + num_std * std
    return cutoff


def calculate_percentile_cutoff(df, column, percentile=95):
    """
    Calculate cutoff based on percentile.

    :param df: DataFrame containing the deviations.
    :type df: pd.DataFrame
    :param column: Column name for which to calculate the cutoff.
    :type column: str
    :param percentile: Percentile to use for the cutoff.
    :type percentile: int
    :return: Cutoff value.
    :rtype: float
    """
    cutoff = np.percentile(df[column].dropna(), percentile)
    return cutoff


def filter_by_cutoffs(
    df, length_cutoff=1.0, angle_cutoff=1.0, ca_cutoff=8.0, minimum_distance=2.0
):
    """
    Filter the DataFrame based on distance, angle, and Ca distance cutoffs. Ca cutoff
    dominates the filter and will override the distance and angle cutoffs. Note: The
    filter is applied if the Ca distance is less than or equal to 2.0 Angstroms, since
    this is physically impossible.

    :param df: DataFrame containing the deviations.
    :type df: pd.DataFrame
    :param length_cutoff: Cutoff value for Bond Length Deviation.
    :type distance_cutoff: float
    :param angle_cutoff: Cutoff value for angle deviation.
    :type angle_cutoff: float
    :param ca_cutoff: Cutoff value for Ca distance.
    :type ca_cutoff: float
    :return: Filtered DataFrame.
    :rtype: pd.DataFrame
    """
    filtered_df = df[
        (df["Bondlength_Deviation"] <= length_cutoff)
        & (df["Angle_Deviation"] <= angle_cutoff)
        & (df["Ca_Distance"] > minimum_distance)
        & (df["Ca_Distance"] < ca_cutoff)
    ]
    return filtered_df


def bad_filter_by_cutoffs(
    df, distance_cutoff=1.0, angle_cutoff=1.0, ca_cutoff=8.0, minimum_distance=2.0
):
    """
    Return the DataFrame objects that are GREATER than the cutoff based on distance,
    angle, and Ca distance cutoffs. Used to get the bad structures. Ca cutoff
    dominates the filter and will override the distance and angle cutoffs.

    :param df: DataFrame containing the deviations.
    :type df: pd.DataFrame
    :param distance_cutoff: Cutoff value for Bond Length Deviation.
    :type distance_cutoff: float
    :param angle_cutoff: Cutoff value for angle deviation.
    :type angle_cutoff: float
    :param ca_cutoff: Cutoff value for Ca distance.
    :type ca_cutoff: float
    :return: Filtered DataFrame.
    :rtype: pd.DataFrame
    """
    filtered_df = df[
        (df["Bondlength_Deviation"] > distance_cutoff)
        & (df["Angle_Deviation"] > angle_cutoff)
        & (df["Ca_Distance"] > ca_cutoff)
        & (df["Ca_Distance"] < minimum_distance)
    ]
    return filtered_df


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
