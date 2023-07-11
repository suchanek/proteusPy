'''
Utility functions for the proteusPy package \n
Author: Eric G. Suchanek, PhD. \n
License: MIT\n
Copyright (c)2023 Eric G. Suchanek, PhD, all rights reserved
'''

# Last modification 2/18/23 -egs-

__pdoc__ = {'__all__': True}

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import itertools

import copy
import subprocess
import pandas as pd

from numpy import linspace
from matplotlib import cm

import proteusPy
from proteusPy.ProteusPyWarning import ProteusPyWarning

from Bio.PDB.vectors import Vector
from Bio.PDB import PDBParser

from proteusPy.data import DATA_DIR

def distance_squared(p1: np.array, p2: np.array) -> np.array:
    '''
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
    '''
    return np.sum(np.square(np.subtract(p1, p2)))

def distance3d(p1: Vector, p2: Vector) -> float:
    '''
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
    '''

    _p1 = p1.get_array()
    _p2 = p2.get_array()
    if (len(_p1) != 3 or len(_p2) != 3):
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
               [  0, 127, 255],
               [123, 255, 123],
               [255, 151,   0],
               [127,   0,   0]], dtype=uint8)
    """
    norm = np.linspace(0.0, 1.0, steps)
    colormap = cm.get_cmap('jet', steps)
    rgbcol = colormap(norm, bytes=True)[:,:3]

    return rgbcol

def grid_dimensions(n: int) -> tuple:
    '''
    Computes the number of rows and columns needed to display a list of length `n`.
    
    Args:
        n (int): Length of input list
    
    Returns:
        tuple: Number of rows and columns required to display input list
    '''
    if n == 1:
        return 1, 1
    elif n == 2:
        return 1, 2
    else:
        root = math.sqrt(n)
        cols = math.ceil(root)
        rows = cols - 1 if cols * (cols - 1) >= n else cols
        return rows, cols
 
def Ogrid_dimensions(n):
    '''
    Calculate rows and columns for the given needed to display
    a given number of disulfides in a square aspect.

    :param n: Number of Disulfides
    :return: int rows, columns
    '''
    
    res = math.ceil(math.sqrt(n))
    return res, res

def Check_chains(pdbid, pdbdir, verbose=True):
    '''
    Returns True if structure has multiple chains of identical length,
    False otherwise. Primarily internal use.

    :param pdbid: PDBID identifier
    :param pdbdir: PDB directory containing structures
    :param verbose: Verbosity, defaults to True
    '''
    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(pdbid, file=f'{pdbdir}pdb{pdbid}.ent')
    
    # dictionary of tuples with SSBond prox and distal
    ssbond_dict = structure.header['ssbond']
    
    if verbose:
        print(f'ssbond dict: {ssbond_dict}')

    same = False
    model = structure[0]
    chainlist = model.get_list()

    if len(chainlist) > 1:
        chain_lens = []
        if verbose:
            print(f'multiple chains. {chainlist}')
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                print(f'Chain: {chain_id}, length: {chain_length}')
            chain_lens.append(chain_length)

        if np.min(chain_lens) != np.max(chain_lens):
            same = False
            if verbose:
                print(f'chain lengths are unequal: {chain_lens}')
        else:
            same = True
            if verbose:
                print(f'Chains are equal length, assuming the same. {chain_lens}')
    return(same)

# given the full dictionary, walk through all the keys (PDB ID)
# for each PDB_ID SS list, find and extract the SS for the first chain
# update the 'pruned' dict with the now shorter SS list

def extract_firstchain_ss(sslist, verbose=False):
    '''
    Function extracts disulfides from the first chain found in
    the SSdict, returns them as a DisulfideList along with the
    number of Xchain disulfides.

    :param sslist: Starting SS list
    :return: (Pruned SS list, xchain) 
    '''

    chainlist = []
    pc = dc = ''
    res = proteusPy.DisulfideList.DisulfideList([], sslist.id)
    xchain = 0

    # build list of chains
    for ss in sslist:
        pc = ss.proximal_chain
        dc = ss.distal_chain
        if pc != dc:
            xchain += 1
            if verbose:
                print(f'--> extract_firstchain_ss(): Cross chain ss: {ss}')
        chainlist.append(pc)
    chain = chainlist[0]

    for ss in sslist:
        if ss.proximal_chain == chain:
            res.append(ss)
    
    return res, xchain

def prune_extra_ss(sslist):
    '''
    Given a dict of disulfides, check for extra chains, grab only the disulfides from
    the first chain and return a dict containing only the first chain disulfides

    :param ssdict: input dictionary with disulfides
    '''
    xchain = 0

    #print(f'Processing: {ss} with: {sslist}')
    id = sslist.pdb_id
    pruned_list = proteusPy.DisulfideList.DisulfideList([], id)
    pruned_list, xchain = extract_firstchain_ss(sslist)
        
    return copy.deepcopy(pruned_list), xchain

def download_file(url, directory):
    '''
    Download the given URL to the input directory

    :param url: File URL
    :param directory: Directory path for saving.
    '''
    file_name = url.split("/")[-1]
    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        command = ["wget", "-P", directory, url]
        subprocess.run(command, check=True)
        print("Download complete.")
    else:
        print(f"{file_name} already exists in {directory}.")

import psutil

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def print_memory_used():
    '''
    Print memory used by the proteusPy process (GB).
    '''
    mem = get_memory_usage() / (1024 ** 3) # to GB

    print(f'proteusPy {proteusPy.__version__}: Memory Used: {mem:.2f} GB')

def image_to_ascii_art(fname, nwidth):
    '''
    Convert an image to ASCII art of given text width.

    Function takes an input filename and width and prints an ASCII art representation to console.

    :param fname: Input filename.
    :param nwidth: Output width in characters.
    '''
    from PIL import Image
    from sklearn.preprocessing import minmax_scale

    # Open the image file
    image = Image.open(fname)

    # Resize the image to reduce the number of pixels
    width, height = image.size
    aspect_ratio = height/width
    new_width = nwidth
    new_height = aspect_ratio * new_width * 0.55  # 0.55 is an adjustment factor
    image = image.resize((new_width, int(new_height)))

    # Convert the image to grayscale.
    image = image.convert("L")

    # Define the ASCII character set to use (inverted colormap)
    char_set = ["@", "#", "8", "&", "o", ":", "*", ".", " "]

    # Normalize the pixel values in the image.
    pixel_data = list(image.getdata())
    pixel_data_norm = minmax_scale(pixel_data, feature_range=(0, len(char_set)-1), copy=True)
    pixel_data_norm = [int(x) for x in pixel_data_norm]

    char_array = [char_set[pixel] for pixel in pixel_data_norm]
    # Generate the ASCII art string
    ascii_art = "\n".join(["".join([char_array[i+j] for j in range(new_width)]) for i in range(0, len(char_array), new_width)])

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
    import pandas as pd
    import itertools

    if base == 2:
        states = ['-', '+']
    elif base == 3:
        states = ['-', '+', '*']
    elif base == 4:
        states = ['-', '+', '*', '@']
    else:
        raise ValueError("Unsupported base")

    combinations = list(itertools.product(states, repeat=5))
    df = pd.DataFrame(combinations, columns=['chi1', 'chi2', 'chi3', 'chi4', 'chi5'])
    return df

def sort_by_column(df, column):
    """
    Sorts a Pandas DataFrame by the values in the 'incidence' column in descending order.

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
    import matplotlib.pyplot as plt
    import numpy as np
    
    from proteusPy.angle_annotation import AngleAnnotation

    # Helper function to draw angle easily.
    def plot_angle(ax, pos, angle, length=0.95, acol="C0", **kwargs):
        vec2 = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        xy = np.c_[[length, 0], [0, 0], vec2*length].T + np.array(pos)
        ax.plot(*xy.T, color=acol)
        return AngleAnnotation(pos, xy[0], xy[2], ax=ax, **kwargs)

    #fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig, ax1= plt.subplots(sharex=True)

    #ax1, ax2 = fig.subplots(1, 2, sharey=True, sharex=True)

    fig.suptitle("SS Torsion Classes")
    fig.set_dpi(220)
    fig.set_size_inches(6, 6)

    fig.canvas.draw()  # Need to draw the figure to define renderer

    # Showcase different text positions.
    ax1.margins(y=0.4)
    ax1.set_title("textposition")
    _text = f"${360/classes}°$"
    kw = dict(size=75, unit="points", text=_text)

    #am6 = plot_angle(ax1, (2.0, 0), 60, textposition="inside", **kw)
    am7 = plot_angle(ax1, (0, 0), 360/classes, textposition="outside", **kw)

    # Create a list of segment values
    # !!!
    values = [1 for _ in range(classes)]

    # Create the pie chart
    #fig, ax = plt.subplots()
    wedges, _ = ax1.pie(
        values, startangle=0, counterclock=False, wedgeprops=dict(width=0.65))

    # Set the chart title and size
    ax1.set_title(f'{classes}-Class Angular Layout')

    # Set the segment colors
    color_palette = plt.cm.get_cmap('tab20', classes)
    ax1.set_prop_cycle('color', [color_palette(i) for i in range(classes)])

    # Create the legend
    legend_labels = [f'Class {i+1}' for i in range(classes)]
    legend = ax1.legend(wedges, legend_labels, title='Classes', loc='center left', bbox_to_anchor=(1.2, 0.5))

    # Set the legend fontsize
    plt.setp(legend.get_title(), fontsize='large')
    plt.setp(legend.get_texts(), fontsize='medium')

    # Show the chart
    fig.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
