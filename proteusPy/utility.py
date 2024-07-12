"""
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: BSD\n
Copyright (c)2024 Eric G. Suchanek, PhD, all rights reserved
"""

# Last modification 7/13/24 -egs-


import copy
import datetime
import glob
import math
import os
import pickle
import subprocess
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Vector
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from matplotlib import cm

import proteusPy
from proteusPy import DisulfideList, ProteusPyWarning

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

# Ignore PDBConstructionWarning
import warnings

from proteusPy.ProteusGlobals import (
    MODEL_DIR,
    PBAR_COLS,
    PDB_DIR,
    PROBLEM_ID_FILE,
    SS_DICT_PICKLE_FILE,
    SS_ID_FILE,
    SS_PICKLE_FILE,
    SS_TORSIONS_FILE,
)

warnings.simplefilter("ignore", PDBConstructionWarning)


def distance_squared(p1: np.array, p2: np.array) -> np.array:
    """
    Return the square of the N-dimensional distance between the
    two arrays.

    :param np.array p1: N-dimensional array 1
    :param np.array p2: N-dimensional array 2
    :return: np.array N-dimensional distance squared Å^2

    Example
    >>> from proteusPy.utility import distance_squared
    >>> p1 = np.array([1.0, 0.0, 0.0])
    >>> p2 = np.array([0, 1.0, 0])
    >>> distance_squared(p1, p2)
    2.0
    """
    return np.sum(np.square(np.subtract(p1, p2)))


def distance3d(p1: Vector, p2: Vector) -> float:
    """
    Calculate the 3D Euclidean distance for 2 Vector objects

    :param Vector p1: Point1
    :param Vector p2: Point2
    :return float distance: Distance between two points, Å

    Example:
    >>> from proteusPy.utility import distance3d
    >>> p1 = Vector(1, 0, 0)
    >>> p2 = Vector(0, 1, 0)
    >>> distance3d(p1,p2)
    1.4142135623730951
    """

    _p1 = p1.get_array()
    _p2 = p2.get_array()
    if len(_p1) != 3 or len(_p2) != 3:
        raise ProteusPyWarning("distance3d() requires vectors of length 3!")
    d = math.dist(_p1, _p2)
    return d


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


def Check_chains(pdbid, pdbdir, verbose=True):
    """
    Return True if structure has multiple chains of identical length,
    False otherwise. Primarily internal use.

    :param pdbid: PDBID identifier
    :param pdbdir: PDB directory containing structures
    :param verbose: Verbosity, defaults to True
    """
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
            print(f"multiple chains. {chainlist}")
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                print(f"Chain: {chain_id}, length: {chain_length}")
            chain_lens.append(chain_length)

        if np.min(chain_lens) != np.max(chain_lens):
            same = False
            if verbose:
                print(f"chain lengths are unequal: {chain_lens}")
        else:
            same = True
            if verbose:
                print(f"Chains are equal length, assuming the same. {chain_lens}")
    return same


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
                print(f"--> extract_firstchain_ss(): Cross chain ss: {ss}")
        chainlist.append(pc)
    try:
        chain = chainlist[0]
    except IndexError:
        print(f"--> extract_firstchain_ss(): No chains found in SS list: {chain}")
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


def download_file(url, directory):
    """
    Download the given URL to the input directory

    :param url: File URL
    :param directory: Directory path for saving.
    """
    file_name = url.split("/")[-1]
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        command = ["wget", "-P", directory, url]
        subprocess.run(command, check=True)
        print("Download complete.")
    else:
        print(f"{file_name} already exists in {directory}.")


def get_memory_usage():
    import psutil

    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss


def print_memory_used():
    """
    Print memory used by the proteusPy process (GB).
    """
    mem = get_memory_usage() / (1024**3)  # to GB

    print(f"proteusPy {proteusPy.__version__}: Memory Used: {mem:.2f} GB")


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


def parse_ssbond_header_rec(ssbond_dict: dict) -> list:
    """
    Parse the SSBOND dict returned by parse_pdb_header.
    NB: Requires EGS-Modified BIO.parse_pdb_header.py.
    This is used internally.

    :param ssbond_dict: the input SSBOND dict
    :return: A list of tuples representing the proximal,
        distal residue ids for the Disulfide.

    """
    disulfide_list = []
    for ssb in ssbond_dict.items():
        disulfide_list.append(ssb[1])

    return disulfide_list


#
# Function reads a comma separated list of PDB IDs and download the corresponding
# .ent files to the PDB_DIR global.
# Used to download the list of proteins containing at least one SS bond
# with the ID list generated from: http://www.rcsb.org/
#


def Download_Disulfides(
    pdb_home=PDB_DIR, model_home=MODEL_DIR, verbose=False, reset=False
) -> None:
    """
    Read a comma separated list of PDB IDs and download them
    to the pdb_home path.

    This utility function is used to download proteins containing at least
    one SS bond with the ID list generated from: http://www.rcsb.org/.

    This is the primary data loader for the proteusPy Disulfide
    analysis package. The list of IDs represents files in the
    RCSB containing > 1 disulfide bond, and it contains
    over 39000 structures. The total download takes about 12 hours. The
    function keeps track of downloaded files so it's possible to interrupt and
    restart the download without duplicating effort.

    :param pdb_home: Path for downloaded files, defaults to PDB_DIR
    :param model_home: Path for extracted data, defaults to MODEL_DIR
    :param verbose: Verbosity, defaults to False
    :param reset: Reset the downloaded files index. Used to restart the download.
    :raises DisulfideIOException: I/O error raised when the PDB file is not found.
    """
    import os
    import time

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


def remove_duplicate_ss(sslist: DisulfideList) -> DisulfideList:
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
    baddir=PDB_DIR + "/bad/",
    datadir=MODEL_DIR,
    picklefile=SS_PICKLE_FILE,
    torsionfile=SS_TORSIONS_FILE,
    problemfile=PROBLEM_ID_FILE,
    dictfile=SS_DICT_PICKLE_FILE,
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
    :param torsionfile:    Name of the disulfide torsion file .csv created
    :param problemfile:    Name of the .csv file containing problem ids
    :param dictfile:       Name of the .pkl file
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

    from proteusPy import DisulfideList, load_disulfides_from_id
    from proteusPy.Disulfide import Torsion_DF_Cols

    bad_dir = baddir

    entrylist = []
    problem_ids = []
    bad = bad_dist = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_dict = {}
    All_ss_list = DisulfideList([], "PDB_SS")
    All_ss_dict2 = {}  # new dict of pointers to indices

    start = time.time()
    cwd = os.getcwd()

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    os.chdir(pdbdir)

    ss_filelist = glob.glob("*.ent")
    tot = len(ss_filelist)

    if verbose:
        print(f"PDB Directory {pdbdir} contains: {tot} files")

    # the filenames are in the form pdb{entry}.ent, I loop through them and extract
    # the PDB ID, with Disulfide.name_to_id(), then add to entrylist.

    for entry in ss_filelist:
        entrylist.append(name_to_id(entry))

    # create a dataframe with the following columns for the disulfide conformations
    # extracted from the structure

    SS_df = pd.DataFrame(columns=Torsion_DF_Cols)

    # define a tqdm progressbar using the fully loaded entrylist list.
    # If numb is passed then
    # only do the last numb entries.

    if numb > 0:
        pbar = tqdm(entrylist[:numb], ncols=PBAR_COLS)
    else:
        pbar = tqdm(entrylist, ncols=PBAR_COLS)

    tot = 0
    cnt = 0
    # loop over ss_filelist, create disulfides and initialize them
    for entry in pbar:
        pbar.set_postfix(
            {"ID": entry, "Bad": bad, "Ca": bad_dist, "Cnt": tot}
        )  # update the progress bar

        # returns an empty list if none are found.
        _sslist = DisulfideList([], entry)
        _sslist = load_disulfides_from_id(
            entry, model_numb=0, verbose=verbose, quiet=quiet, pdb_dir=pdbdir
        )
        # !!! sslist, xchain = prune_extra_ss(_sslist)
        # sslist = _sslist
        sslist = remove_duplicate_ss(_sslist)
        if len(sslist) > 0:
            sslist2 = []  # list to hold indices for ss_dict2
            for ss in sslist:
                # Ca distance cutoff
                dist = ss.ca_distance
                if dist >= dist_cutoff and dist_cutoff != -1.0:
                    bad_dist += 1
                    continue  ## was continue

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
                SS_df.loc[len(SS_df.index)] = new_row.copy()  # deep copy
                sslist2.append(cnt)
                cnt += 1
                tot += 1

            # All_ss_dict[entry] = sslist
            # print(f'Entry: {entry}. Dict indices: {sslist2}')
            All_ss_dict2[entry] = sslist2
            # print(f'{entry} ss dict adding: {sslist2}')

        else:
            # at this point I really shouldn't have any bad non-parsible file
            bad += 1
            problem_ids.append(entry)
            if prune:
                shutil.copy(f"pdb{entry}.ent", bad_dir)
                # Delete the original file
                os.remove(f"pdb{entry}.ent")

    if bad > 0:
        prob_cols = ["id"]
        problem_df = pandas.DataFrame(columns=prob_cols)
        problem_df["id"] = problem_ids

        print(
            f"-> Extract_Disulfides(): Found and moved: {len(problem_ids)} non-parsable structures."
        )
        print(
            f"-> Extract_Disulfides(): Saving problem IDs to file: {datadir}{problemfile}"
        )

        problem_df.to_csv(f"{datadir}{problemfile}")
    else:
        if verbose:
            print("-> Extract_Disulfides(): No non-parsable structures found.")

    if bad_dist > 0:
        print(f"-> Extract_Disulfides(): Found and ignored: {bad_dist} long SS bonds.")
    else:
        if verbose:
            print("No problems found.")

    # dump the all_ss list of disulfides to a .pkl file. ~520 MB.
    fname = f"{datadir}{picklefile}"
    print(
        f"-> Extract_Disulfides(): Saving {len(All_ss_list)} Disulfides to file: {fname}"
    )

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_list, f)

    # dump the dict2 disulfides to a .pkl file. ~520 MB.
    dict_len = len(All_ss_dict2)
    fname = f"{datadir}{dictfile}"
    print(
        f"-> Extract_Disulfides(): Saving indices of {dict_len} Disulfide-containing PDB IDs to file: {fname}"
    )

    with open(fname, "wb+") as f:
        pickle.dump(All_ss_dict2, f)

    # save the torsions
    fname = f"{datadir}{torsionfile}"
    print(f"-> Extract_Disulfides(): Saving torsions to file: {fname}")
    SS_df.to_csv(fname)

    end = time.time()
    elapsed = end - start

    print(
        f"-> Extract_Disulfides(): Disulfide Extraction complete! Elapsed time:\
    	 {datetime.timedelta(seconds=elapsed)} (h:m:s)"
    )

    # return to original directory
    os.chdir(cwd)
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
        # Check if the filename follows the expected format
        if filename.startswith("pdb") and filename.endswith(".ent"):
            # Extract the ID part of the filename
            return filename[3:-4]
        else:
            raise ValueError(
                "Filename does not follow the expected format 'pdb{id}.ent'"
            )

    cwd = os.getcwd()

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    os.chdir(pdbdir)
    id = extract_id_from_filename(pdbid)

    # returns an empty list if none are found.
    _sslist = DisulfideList([], id)
    _sslist = load_disulfides_from_id(
        id, model_numb=0, verbose=verbose, quiet=quiet, pdb_dir=pdbdir
    )
    # sslist, xchain = prune_extra_ss(_sslist)
    sslist = _sslist

    if len(sslist) == 0:
        print(f"--> Can't parse: {pdbid}")

    # return to original directory
    os.chdir(cwd)
    return sslist


def check_header_from_file(
    filename: str, model_numb=0, verbose=False, dbg=False
) -> bool:
    """
    Parse the Disulfides contained in the PDB file.

    NB: Requires EGS-Modified BIO.parse_pdb_header.py from https://github.com/suchanek/biopython/

    :param filename: Filename for the entry.
    :param model_numb: Model number to use, defaults to 0 for single structure files.
    :param verbose: Print info while parsing
    :return: True if parseable

    Example:
      Assuming ```DATA_DIR``` has the pdb5rsa.ent file (it should!), we can load the disulfides
      with the following:

    >>> from proteusPy import check_header_from_file, DATA_DIR
    >>> OK = False
    >>> OK = check_header_from_file(f'{DATA_DIR}pdb5rsa.ent', verbose=False)
    >>> OK
    True
    """
    import os

    i = 1
    proximal = distal = -1
    _chaina = None
    _chainb = None

    parser = PDBParser(PERMISSIVE=True)

    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    try:
        structure = parser.get_structure("tmp", file=filename)
        struct_name = structure.get_id()
        model = structure[model_numb]
    except FileNotFoundError:
        mess = f"Error: The file {filename} does not exist."
        raise DisulfideParseWarning(mess)

    except Exception as e:
        mess = f"An error occurred: {e}"
        raise DisulfideParseWarning(mess)

    if verbose:
        print(f"-> check_header_from_file() - Parsing file: {filename}:")

    ssbond_dict = structure.header["ssbond"]  # NB: this requires the modified code

    # list of tuples with (proximal distal chaina chainb)
    ssbonds = parse_ssbond_header_rec(ssbond_dict)
    if len(ssbonds) == 0:
        if verbose:
            print("-> check_header_from_file(): no bonds found in bondlist.")
        return False

    for pair in ssbonds:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]

        if not proximal.isnumeric() or not distal.isnumeric():
            if verbose:
                mess = f" ! Cannot parse SSBond record (non-numeric IDs):\
                 {struct_name} Prox:  {proximal} {chain1_id} Dist: {distal} {chain2_id}"
                warnings.warn(mess, DisulfideParseWarning)
            continue  # was pass
        else:
            proximal = int(proximal)
            distal = int(distal)

        chain1_id = pair[2]
        chain2_id = pair[3]

        _chaina = model[chain1_id]
        _chainb = model[chain2_id]

        if chain1_id != chain2_id:
            if verbose:
                mess = f" -> Cross Chain SS for: Prox: {proximal}{chain1_id} Dist: {distal}{chain2_id}"
                warnings.warn(mess, DisulfideParseWarning)
                pass  # was break

        try:
            prox_res = _chaina[proximal]
            dist_res = _chainb[distal]
        except KeyError:
            print(
                f" ! Cannot parse SSBond record (KeyError): {struct_name} Prox: <{proximal}> {chain1_id} Dist: <{distal}> {chain2_id}"
            )
            return False

        if (_chaina is not None) and (_chainb is not None):
            if _chaina[proximal].is_disordered() or _chainb[distal].is_disordered():
                continue
            else:
                if verbose:
                    print(
                        f" -> SSBond: {i}: {struct_name}: {proximal}{chain1_id} - {distal}{chain2_id}"
                    )
        else:
            if dbg:
                print(
                    f" -> NULL chain(s): {struct_name}: {proximal}{chain1_id} - {distal}{chain2_id}"
                )
        i += 1
    return True


def check_header_from_id(
    struct_name: str, pdb_dir=".", model_numb=0, verbose=False, dbg=False
) -> bool:
    """
    Check parsability PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in ```MODEL_DIR``` path.

    NB: Requires EGS-Modified BIO.parse_pdb_header.py from https://github.com/suchanek/biopython/

    :param struct_name: the name of the PDB entry.
    :param pdb_dir: path to the PDB files, defaults to PDB_DIR
    :param model_numb: model number to use, defaults to 0 for single structure files.
    :param verbose: print info while parsing
    :param dbg: Debugging Flag
    :return: True if OK, False otherwise

    Example:
      Assuming the DATA_DIR has the pdb5rsa.ent file we can check the file thusly:
      (assumes the PDB environment variable is set to the PDB directory.)

    >>> import os
    >>> from proteusPy import Disulfide, check_header_from_id, DATA_DIR
    >>> OK = False
    >>> OK = check_header_from_id('5rsa', pdb_dir=DATA_DIR, verbose=True)
     -> SSBond: 1: 5rsa: 26A - 84A
     -> SSBond: 2: 5rsa: 40A - 95A
     -> SSBond: 3: 5rsa: 58A - 110A
     -> SSBond: 4: 5rsa: 65A - 72A
    >>> OK
    True
    """
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(struct_name, file=f"{pdb_dir}pdb{struct_name}.ent")
    model = structure[0]

    ssbond_dict = structure.header["ssbond"]  # NB: this requires the modified code

    bondlist = []
    i = 0

    # get a list of tuples containing the proximal, distal residue IDs for
    # all SSBonds in the chain.
    bondlist = parse_ssbond_header_rec(ssbond_dict)

    if len(bondlist) == 0:
        if verbose:
            print("-> check_header_from_id(): no bonds found in bondlist.")
        return False

    for pair in bondlist:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]
        chain1 = pair[2]
        chain2 = pair[3]

        chaina = model[chain1]
        chainb = model[chain2]

        try:
            prox_residue = chaina[proximal]
            dist_residue = chainb[distal]

            prox_residue.disordered_select("CYS")
            dist_residue.disordered_select("CYS")

            if (
                prox_residue.get_resname() != "CYS"
                or dist_residue.get_resname() != "CYS"
            ):
                if verbose:
                    print(
                        f"build_disulfide() requires CYS at both residues:\
                     {prox_residue.get_resname()} {dist_residue.get_resname()}"
                    )
                return False
        except KeyError:
            if dbg:
                print(
                    f"Keyerror: {struct_name}: {proximal} {chain1} - {distal} {chain2}"
                )
                return False

        if verbose:
            print(
                f" -> SSBond: {i+1}: {struct_name}: {proximal}{chain1} - {distal}{chain2}"
            )

        i += 1
    return True


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# end of file
