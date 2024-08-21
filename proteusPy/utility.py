"""
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: BSD\n
Copyright (c)2024 Eric G. Suchanek, PhD, all rights reserved
"""

# Last modification 7/25/24 -egs-


import copy
import datetime
import glob
import logging
import math
import os
import pickle
import subprocess
import time
from pathlib import Path

import psutil

from proteusPy import Disulfide
from proteusPy.logger_config import get_logger

_logger = get_logger(__name__)

# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from proteusPy import DisulfideList, __version__
from proteusPy.DisulfideExceptions import DisulfideIOException, DisulfideParseWarning
from proteusPy.ProteusPyWarning import ProteusPyWarning
from proteusPy.vector3D import Vector3D

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__  # type: ignore
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

from tqdm.contrib.logging import logging_redirect_tqdm

from proteusPy.ProteusGlobals import (
    DATA_DIR,
    MODEL_DIR,
    PDB_DIR,
    PROBLEM_ID_FILE,
    SS_ID_FILE,
    SS_PICKLE_FILE,
)

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
    Return an array of uniformly spaced RGB values using the 'jet' colormap.

    :param steps: The number of steps in the output array.

    :return: An array of uniformly spaced RGB values using the 'jet' colormap. The shape of the array is (steps, 3).
    :rtype: numpy.ndarray

    Example:
        >>> get_jet_colormap(5)
        array([[  0,   0, 127],
               [  0, 128, 255],
               [124, 255, 121],
               [255, 148,   0],
               [127,   0,   0]], dtype=uint8)
    """
    import matplotlib

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


'''
# This function will be deprecated in the future.
def Check_chains(pdbid, pdbdir, verbose=True) -> bool:
    """
    Return True if structure has multiple chains of identical length,
    False otherwise. Primarily internal use.

    :param pdbid: PDBID identifier
    :param pdbdir: PDB directory containing structures
    :param verbose: Verbosity, defaults to True
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(pdbid, file=f"{pdbdir}pdb{pdbid}.ent")

    # dictionary of tuples with SSBond prox and distal
    ssbond_dict = structure.header["ssbond"]

    if verbose:
        print(f"ssbond dict: {ssbond_dict}")

    same = False
    model = structure[0]
    chainlist = model.get_list()

    if len(chainlist) > 1:
        chain_lens = []
        if verbose:
            _logger.info(f"multiple chains. {chainlist}")
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                _logger.info(f"Chain: {chain_id}, length: {chain_length}")
            chain_lens.append(chain_length)

        if np.min(chain_lens) != np.max(chain_lens):
            same = False
            if verbose:
                _logger.warning(f"chain lengths are unequal: {chain_lens}")
        else:
            same = True
            if verbose:
                _logger.info(
                    f"Chains are equal length, assuming the same. {chain_lens}"
                )
    return same

'''

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
    from proteusPy import DisulfideList

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
                _logger.info(f"extract_firstchain_ss(): Cross chain ss: {ss}")
        chainlist.append(pc)
    try:
        chain = chainlist[0]
    except IndexError:
        _logger.warning(f"extract_firstchain_ss(): No chains found in SS list: {chain}")
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
    from proteusPy import DisulfideList

    # print(f'Processing: {ss} with: {sslist}')
    id = sslist.pdb_id
    pruned_list = DisulfideList([], id)
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
            _logger.info(f"Downloading {file_name}...")
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
    from sklearn.preprocessing import minmax_scale

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


def generate_vector_dataframe(base=3):
    """
    Generate a pandas DataFrame containing all combinations for a vector of length 5 with a given base.

    :param base: An integer representing the base of the vector elements. Must be 2, 3, or 4.
    :return: A pandas DataFrame with columns 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', where each row
             contains all combinations for a vector of length 5 with the specified base. The symbols used
             to represent the vector elements are '-' and '+' for base 2, '-' '+' and '*' for base 3,
             and '-' '+' '*' and '@' for base 4.
    :raises ValueError: If the specified base is not supported (i.e., not 2, 3, or 4).
    """
    import itertools

    import pandas as pd

    if base == 2:
        states = ["-", "+"]
    elif base == 3:
        states = ["-", "+", "*"]
    elif base == 4:
        states = ["-", "+", "*", "@"]
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


def plot_class_chart(classes: int) -> None:
    """
    Create a Matplotlib pie chart with `classes` segments of equal size.

    Parameters:
        classes (int): The number of segments to create in the pie chart.

    Returns:
        None

    Example:
    >>> plot_class_chart(4)

    This will create a pie chart with 4 equal segments.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    from proteusPy.angle_annotation import AngleAnnotation

    # Helper function to draw angle easily.
    def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
        vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        xy = np.c_[[length, 0], [0, 0], vec2 * length].T + np.array(pos)
        ax.plot(*xy.T, color=acol)
        return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

    # fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig, ax1 = plt.subplots(sharex=True)

    # ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)

    fig.suptitle("SS Torsion Classes")
    fig.set_dpi(220)
    fig.set_size_inches(6, 6)

    fig.canvas.draw()  # Need to draw the figure to define renderer

    # Showcase different text positions.
    ax1.margins(y=0.4)
    ax1.set_title("textposition")
    _text = f"${360/classes}°$"
    kw = dict(size=75, unit="points", text=_text)

    # am6 = plot_angle(ax1, (2.0, 0), 60, textposition="inside", **kw)
    am7 = plot_angle(ax1, (0, 0), 360 / classes, textposition="outside", **kw)

    # Create a list of segment values
    # !!!
    values = [1 for _ in range(classes)]

    # Create the pie chart
    # fig, ax = plt.subplots()
    wedges, _ = ax1.pie(
        values, startangle=0, counterclock=False, wedgeprops=dict(width=0.65)
    )

    # Set the chart title and size
    ax1.set_title(f"{classes}-Class Angular Layout")

    # Set the segment colors
    color_palette = matplotlib.colormaps["tab20"]

    # color_palette = plt.cm.get_cmap("tab20", classes)
    ax1.set_prop_cycle("color", [color_palette(i) for i in range(classes)])

    # Create the legend
    legend_labels = [f"Class {i+1}" for i in range(classes)]
    legend = ax1.legend(
        wedges,
        legend_labels,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
    )

    # Set the legend fontsize
    plt.setp(legend.get_title(), fontsize="large")
    plt.setp(legend.get_texts(), fontsize="medium")

    # Show the chart
    fig.show()


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

    response = requests.post(batch_url, headers=headers, json=data)
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
        pm(f"show spheres, solvent")

    pm(f"color green, resi {proximal}+{distal}")
    pm(f"zoom resi {proximal}+{distal}")
    if ray:
        pm(f"ray {width}, {height}")
        pm(f"png {fname}, dpi={dpi}")

    input("Press Enter to continue...")
    return None


def parse_ssbond_header_rec(ssbond_dict: dict, verbose=False) -> list:
    """
    Parse the SSBOND dict returned by parse_pdb_header.
    NB: Requires EGS-Modified BIO.parse_pdb_header.py.
    This is used internally.

    :param ssbond_dict: the input SSBOND dict
    :return: A list of tuples representing the proximal,
        distal residue ids for the Disulfide.

    """
    if verbose:
        print(f"parse_ssbond_header_rec(): {ssbond_dict}")

    disulfide_list = []
    for ssb in ssbond_dict.items():
        disulfide_list.append(ssb[1])

    return disulfide_list


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
    import os
    import time

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
        ssfile = open(ssfilename)
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
        SS_DONE = []
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
    pruned = []
    for ss in sslist:
        if ss not in pruned:
            pruned.append(ss)
    return pruned


# Function extracts the disulfide bonds from the PDB files and creates the .pkl files
# needed for the proteusPy.DisulfideLoader.DisulfideLoader class.


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

    def name_to_id(fname: str) -> str:
        """
        Returns the PDB ID from the filename.

        :param fname: Complete PDB filename
        :return: PDB ID
        """
        ent = fname[3:-4]
        return ent

    import shutil

    from proteusPy import DisulfideList, load_disulfides_from_id
    from proteusPy.Disulfide import Torsion_DF_Cols

    if quiet:
        _logger.setLevel(logging.ERROR)

    bad_dir = baddir

    entrylist = []
    sslist = []
    problem_ids = []
    bad = bad_dist = tot = cnt = 0

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
        _logger.info(f"PDB Ids: {entrylist}, len: {len(entrylist)}")

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
            pbar = tqdm(entrylist[:numb], ncols=PBAR_COLS)
        else:
            pbar = tqdm(entrylist, ncols=PBAR_COLS)

        for entry in pbar:
            _sslist = DisulfideList([], entry)

            # returns an empty list if none are found.
            _sslist = load_disulfides_from_id(
                entry,
                model_numb=0,
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
                    new_row = [
                        ss.pdb_id,
                        ss.name,
                        ss.proximal,
                        ss.distal,
                        ss.chi1,
                        ss.chi2,
                        ss.chi3,
                        ss.chi4,
                        ss.chi5,
                        ss.energy,
                        ss.ca_distance,
                        ss.cb_distance,
                        ss.phiprox,
                        ss.psiprox,
                        ss.phidist,
                        ss.psidist,
                        ss.torsion_length,
                        ss.rho,
                    ]

                    # add the row to the end of the dataframe
                    # SS_df.loc[len(SS_df.index)] = new_row.copy()  # deep copy
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
                    _logger.warning(f"Extract_Disulfides(): No SS parsed for: {entry}!")

                if prune:
                    fname = f"pdb{entry}.ent"
                    # Construct the full path for the new destination file
                    destination_file_path = Path(bad_dir) / fname

                    # Copy the file to the new destination with the correct filename
                    _logger.warning(
                        f"Extract_Disulfides(): Moving {fname} to {destination_file_path}"
                    )
                    shutil.move(fname, destination_file_path)
                continue  ## this entry has no SS bonds, so we break the loop and move on to the next entry

            pbar.set_postfix({"ID": entry, "NoSS": bad, "Cnt": cnt})

    pbar.close()

    if bad > 0:
        prob_cols = ["id"]
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df["id"] = problem_ids

        _logger.warning(
            (
                f"Found and moved: {len(problem_ids)} non-parsable structures."
                f"Saving problem IDs to file: {Path(datadir) / problemfile}"
            )
        )

        problem_df.to_csv(Path(datadir) / problemfile)
    else:  ## no bad files found
        if verbose:
            _logger.info("No non-parsable structures found.")

    if bad_dist > 0:
        if verbose:
            _logger.warning(f"Found and ignored: {bad_dist} long SS bonds.")

    else:
        if verbose:
            _logger.info("Extract_Disulfides(): No problems found.")

    # dump the all_ss list of disulfides to a .pkl file. ~520 MB.
    fname = Path(datadir) / picklefile

    if verbose:
        _logger.info(f"Saving {len(All_ss_list)} Disulfides to file: {fname}")

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_list, f)

    end = time.time()
    elapsed = end - start

    if verbose:
        _logger.info(
            f"Disulfide Extraction complete! Elapsed time:\
    	    {datetime.timedelta(seconds=elapsed)} (h:m:s)"
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
    sslist=[],
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

    import shutil

    from proteusPy import DisulfideList, load_disulfides_from_id
    from proteusPy.Disulfide import Torsion_DF_Cols

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
            f"Extract_Disulfides(): PDB Ids: {entrylist}, len: {len(entrylist)}"
        )

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
            pbar = tqdm(entrylist[:numb], ncols=PBAR_COLS)
        else:
            pbar = tqdm(entrylist, ncols=PBAR_COLS)

        for entry in pbar:
            _sslist = DisulfideList([], entry)

            # returns an empty list if none are found.
            _sslist = load_disulfides_from_id(
                entry,
                model_numb=0,
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
                    new_row = [
                        ss.pdb_id,
                        ss.name,
                        ss.proximal,
                        ss.distal,
                        ss.chi1,
                        ss.chi2,
                        ss.chi3,
                        ss.chi4,
                        ss.chi5,
                        ss.energy,
                        ss.ca_distance,
                        ss.cb_distance,
                        ss.phiprox,
                        ss.psiprox,
                        ss.phidist,
                        ss.psidist,
                        ss.torsion_length,
                        ss.rho,
                    ]

                    # add the row to the end of the dataframe
                    # SS_df.loc[len(SS_df.index)] = new_row.copy()  # deep copy
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
                    _logger.warning(f"Extract_Disulfides(): No SS parsed for: {entry}!")
                if prune:
                    fname = f"pdb{entry}.ent"
                    # Construct the full path for the new destination file
                    destination_file_path = Path(bad_dir) / fname
                    # Copy the file to the new destination with the correct filename
                    _logger.warning(
                        f"Extract_Disulfides(): Moving {fname} to {destination_file_path}"
                    )
                    shutil.move(fname, destination_file_path)
                continue  ## this entry has no SS bonds, so we break the loop and move on to the next entry

            pbar.set_postfix(
                {"ID": entry, "NoSS": bad, "Cnt": tot}
            )  # update the progress bar

    if bad > 0:
        prob_cols = ["id"]
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df["id"] = problem_ids

        _logger.warning(
            (
                f"Found and moved: {len(problem_ids)} non-parsable structures."
                f"Saving problem IDs to file: {datadir}{problemfile}"
            )
        )

        problem_df.to_csv(Path(datadir / problemfile))

    else:  ## no bad files found
        if verbose:
            _logger.info("Extract_Disulfides(): No non-parsable structures found.")

    if bad_dist > 0:
        if verbose:
            _logger.warning(f"Found and ignored: {bad_dist} long SS bonds.")

    else:
        if verbose:
            _logger.info("No problems found.")

    # dump the all_ss list of disulfides to a .pkl file. ~520 MB.
    fname = Path(datadir) / picklefile

    if verbose:
        _logger.info(f"Saving {len(All_ss_list)} Disulfides to file: {fname}")

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_list, f)

    end = time.time()
    elapsed = end - start

    if verbose:
        _logger.info(
            f"Disulfide Extraction complete! Elapsed time:\
    	    {datetime.timedelta(seconds=elapsed)} (h:m:s)"
        )

    # return to original directory
    os.chdir(cwd)

    # restore the logger level
    if quiet:
        _logger.setLevel(logging.WARNING)
    return


def Extract_Disulfide(
    pdbid: str, verbose=False, quiet=True, pdbdir=PDB_DIR, xtra=True
) -> DisulfideList:
    """
    Read the PDB file represented by `pdbid` and return a ``DisulfideList``
    containing the Disulfide bonds found.

    :param verbose:        Display more messages
    :param quiet:          Turn off DisulfideConstruction warnings
    :param pdbdir:         path to PDB files
    :param xtra:           Prune duplicate disulfides
    """

    import glob
    import os
    import shutil
    import time

    import proteusPy
    from proteusPy import DisulfideList, load_disulfides_from_id

    def extract_id_from_filename(filename: str) -> str:
        """
        Extract the ID from a filename formatted as 'pdb{id}.ent'.

        Parameters:
        - filename (str): The filename to extract the ID from.

        Returns:
        - str: The extracted ID.
        """
        basename = os.path.basename(filename)
        # Check if the filename follows the expected format
        if basename.startswith("pdb") and filename.endswith(".ent"):
            # Extract the ID part of the filename
            return filename[3:-4]
        else:
            mess = (
                f"Filename {filename} does not follow the expected format 'pdbid .ent'"
            )
            raise ValueError(mess)

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    id = extract_id_from_filename(pdbid)

    # returns an empty list if none are found.
    _sslist = DisulfideList([], id)
    _sslist = load_disulfides_from_id(
        id, model_numb=0, verbose=verbose, quiet=quiet, pdb_dir=pdbdir
    )

    if len(_sslist) == 0 or _sslist is None:
        print(f"--> Can't find SSBonds: {pdbid}")
        return DisulfideList([], id)

    # return to original directory
    return _sslist


def get_macos_theme():
    """
    Determine the display theme for a macOS computer using AppleScript.

    Returns:
    :return str: 'light', 'dark', or 'none' based on the current display theme.

    Example:
    >>> get_macos_theme()
    'dark'
    """
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
        )

        # Check the output
        if result.returncode == 0:
            theme = result.stdout.strip().lower()
            if theme in ["dark", "light"]:
                return theme
            else:
                return "none"
        else:
            return "none"
    except Exception as e:
        # In case of any exception, return 'none'
        return "none"


# functions to calculate statistics and filter disulfide lists via pandas


def create_deviation_dataframe(disulfide_list):
    """
    Create a DataFrame with columns PDB_ID, SS_Name, Angle_Deviation, Distance_Deviation, Ca Distance
    from a list of disulfides.

    :param disulfide_list: List of disulfide objects.
    :type proteusPy.DisulfideList: list
    :return: DataFrame containing the disulfide information.
    :rtype: pd.DataFrame
    """
    data = {
        "PDB_ID": [],
        "Resolution": [],
        "SS_Name": [],
        "Angle_Deviation": [],
        "Bondlength_Deviation": [],
        "Ca_Distance": [],
    }

    for ss in tqdm(disulfide_list, desc="Processing..."):
        pdb_id = ss.pdb_id
        resolution = ss.resolution
        ca_distance = ss.ca_distance
        angle_deviation = ss.bond_angle_ideality
        distance_deviation = ss.bond_length_ideality

        data["PDB_ID"].append(pdb_id)
        data["Resolution"].append(resolution)
        data["SS_Name"].append(ss.name)
        data["Angle_Deviation"].append(angle_deviation)
        data["Bondlength_Deviation"].append(distance_deviation)
        data["Ca_Distance"].append(ca_distance)

    df = pd.DataFrame(data)
    return df


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


def set_logger_level_for_module(pkg_name, level=""):
    """
    Set the logging level for all loggers within a specified package.

    This function iterates through all registered loggers and sets the logging
    level for those that belong to the specified package.

    :param pkg_name: The name of the package for which to set the logging level.
    :type pkg_name: str
    :param level: The logging level to set (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                  If not specified, the logging level will not be changed.
    :type level: str, optional
    :return: A list of logger names that were found and had their levels set.
    :rtype: list
    """
    logger_dict = logging.Logger.manager.loggerDict
    registered_loggers = [
        name
        for name in logger_dict
        if isinstance(logger_dict[name], logging.Logger) and name.startswith(pkg_name)
    ]
    for logger_name in registered_loggers:
        logger = logging.getLogger(logger_name)
        if level:
            logger.setLevel(level)

    return registered_loggers


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
