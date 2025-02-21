"""
This module, *DisulfideBase*, is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures with an emphasis on disulfide bonds.
It represents the core of the current implementation of *proteusPy* and includes the
``Disulfide`` and ``DisulfideList`` classes, which provide a Python object and methods representing
a physical disulfide bond either extracted from the RCSB protein databank or built using the
``proteusPy.Turtle3D`` class.

This work is based on the original C/C++ implementation by Eric G. Suchanek.

Author: Eric G. Suchanek, PhD
Last Modification: 2025-02-21 16:33:47
"""

# pylint: disable=W1203 # use of print
# pylint: disable=C0103 # invalid name
# pylint: disable=C0301 # line too long
# pylint: disable=C0302 # too many lines in module
# pylint: disable=W0212 # access to protected member
# pylint: disable=W0237 # renaming index
# pylint: disable=W0613 # unused argument
# pylint: disable=C2801 # no dunder method


# Cα N, Cα, Cβ, C', Sγ Å ° ρ

import copy
import logging
import math
from collections import UserList
from itertools import combinations
from math import cos

import numpy as np
from scipy.optimize import minimize

from proteusPy.DisulfideClassManager import DisulfideClassManager
from proteusPy.DisulfideExceptions import (
    DisulfideConstructionException,
    DisulfideConstructionWarning,
    ProteusPyWarning,
)
from proteusPy.DisulfideStats import DisulfideStats
from proteusPy.DisulfideVisualization import DisulfideVisualization
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import _ANG_INIT, _FLOAT_INIT, WINSIZE
from proteusPy.Residue import build_residue
from proteusPy.turtle3D import ORIENT_SIDECHAIN, Turtle3D
from proteusPy.utility import set_pyvista_theme
from proteusPy.vector3D import Vector3D, calc_dihedral, distance3d

np.set_printoptions(suppress=True)

set_pyvista_theme("auto")

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Suppress findfont debug messages
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

ORIGIN = Vector3D(0.0, 0.0, 0.0)

_logger = create_logger(__name__)
_logger.setLevel(logging.ERROR)


class DisulfideList(UserList):
    """
    The class provides a sortable list for Disulfide objects.
    Indexing and slicing are supported, as well as typical list operations like
    ``.insert()``, ``.append()`` and ``.extend().`` The DisulfideList object must be initialized
    with an iterable (tuple, list) and a name. Sorting is keyed by torsional energy.

    The class can also render Disulfides to a pyVista window using the
    [display()](#DisulfideList.display) and [display_overlay()](#DisulfideList.display_overlay)methods.
    """

    def __init__(self, iterable, pid: str = "nil", res=-1.0, quiet=True, fast=False):
        """
        Initialize the DisulfideList.

        :param iterable: An iterable of disulfide bonds
        :param pid: Name for the list, default is "nil"
        :param res: Resolution, default is -1.0. If -1, the average resolution is used
        :param quiet: If True, suppress output, default is True
        :param fast: If True, enable fast mode, default is False
        """
        super().__init__(self.validate_ss(item) for item in iterable)
        self.pdb_id = pid
        self.quiet = quiet

        if not fast:
            if res == -1:
                self._res = self.average_resolution
            else:
                self._res = res
        else:
            self._res = res

    def __getitem__(self, item):
        """
        Retrieve a disulfide from the list.

        :param item: Index or slice
        :return: Sublist
        :rtype: DisulfideList
        :raises IndexError: Invalid index or slice range outside list bounds
        :raises ValueError: Invalid slice step (zero) or invalid slice range
        """
        if isinstance(item, slice):
            # Get list length
            list_len = len(self.data)

            # Validate step first since it's the only parameter that should raise an error
            if item.step == 0:
                raise ValueError("Slice step cannot be zero")

            # Get normalized indices using Python's built-in slice.indices()
            # This handles negative indices and out-of-bounds values by clamping them
            start, stop, step = item.indices(list_len)

            if start < 0 or step < 0:
                _logger.error("Negative indices or steps are not supported")
                return DisulfideList([], "invalid_start_or_step")

            if stop > list_len:
                _logger.error("Slice range exceeds list bounds")
                stop = max(list_len, stop)
                return DisulfideList([], "invalid_slice")

            # Create range of indices
            indices = range(start, stop, step)

            # Create the sublist
            sublist = [self.data[i] for i in indices]

            # Return empty list for empty slice
            if not sublist:
                return DisulfideList([], "empty_slice")

            # Create name with slice info
            name = (
                sublist[0].pdb_id
                + f"_slice[{indices[0]}:{indices[-1]+1}]_{sublist[-1].pdb_id}"
            )
            return DisulfideList(sublist, name)

        return UserList.__getitem__(self, item)

    def __setitem__(self, index, item):
        self.data[index] = self.validate_ss(item)

    def append(self, item):
        """Append the list with a Disulfide"""
        self.data.append(self.validate_ss(item))

    def extend(self, other):
        """Extend the Disulfide list with another Disulfide"""
        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(self.validate_ss(item) for item in other)

    def insert(self, index, item):
        """Insert a Disulfide into the list at the specified index"""
        self.data.insert(index, self.validate_ss(item))

    def describe(self):
        """
        Prints out relevant attributes of the given disulfideList.

        :param disulfideList: A list of disulfide objects.
        :param list_name: The name of the list.
        """
        name = self.pdb_id
        avg_distance = self.average_ca_distance
        avg_energy = self.average_energy
        avg_resolution = self.average_resolution
        list_length = len(self.data)

        if list_length == 0:
            avg_bondangle = 0
            avg_bondlength = 0
        else:
            total_bondangle = 0
            total_bondlength = 0

            for ss in self.data:
                total_bondangle += ss.bond_angle_ideality
                total_bondlength += ss.bond_length_ideality

            avg_bondangle = total_bondangle / list_length
            avg_bondlength = total_bondlength / list_length

        print(f"DisulfideList: {name}")
        print(f"Length: {list_length}")
        print(f"Average energy: {avg_energy:.2f} kcal/mol")
        print(f"Average CA distance: {avg_distance:.2f} Å")
        print(f"Average Resolution: {avg_resolution:.2f} Å")
        print(f"Bond angle deviation: {avg_bondangle:.2f}°")
        print(f"Bond length deviation: {avg_bondlength:.2f} Å")

    @property
    def length(self):
        """Return the length of the list"""
        return len(self.data)

    @property
    def resolution(self) -> float:
        """Average structure resolution for the given list"""
        return self._res

    @resolution.setter
    def resolution(self, value: float):
        """Set the average structure resolution"""
        if not isinstance(value, float):
            raise TypeError("Resolution must be a float.")
        self._res = value

    @property
    def average_resolution(self) -> float:
        """Compute and return the average structure resolution"""
        resolutions = [ss.resolution for ss in self.data if ss.resolution != -1.0]
        return sum(resolutions) / len(resolutions) if resolutions else -1.0

    @property
    def average_ca_distance(self):
        """Average Cα distance (Å) between all atoms in the list"""
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0
        total_dist = sum(ss.ca_distance for ss in sslist)
        return total_dist / tot

    @property
    def average_distance(self):
        """Average distance (Å) between all atoms in the list"""
        sslist = self.data
        cnt = 1
        total = 0.0
        for ss1 in sslist:
            for ss2 in sslist:
                if ss2 == ss1:
                    continue
                total += ss1.Distance_RMS(ss2)
                cnt += 1
        return total / cnt

    @property
    def average_energy(self):
        """Average energy (kcal/mol) between all atoms in the list"""
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0
        total_energy = sum(ss.energy for ss in sslist)
        return total_energy / tot

    @property
    def average_conformation(self):
        """Average conformation [x1, x2, x3, x4, x5] handling circular angles properly"""
        sslist = self.data
        torsions = np.array([ss.torsion_array for ss in sslist])
        # Calculate circular mean for each torsion angle separately
        avg_torsions = np.array(
            [DisulfideStats.circular_mean(torsions[:, i]) for i in range(5)]
        )
        return avg_torsions

    @property
    def average_torsion_distance(self):
        """Average distance in torsion space (degrees)"""
        sslist = self.data
        total = 0
        cnt = 0
        for ss1, ss2 in combinations(sslist, 2):
            total += ss1.torsion_distance(ss2)
            cnt += 1
        return float(total / cnt) if cnt > 0 else 0

    @property
    def center_of_mass(self):
        """Center of mass for the Disulfide list"""
        sslist = self.data
        tot = len(sslist)
        if tot == 0:
            return 0.0
        total_cofmass = sum(ss.cofmass for ss in sslist)
        return total_cofmass / tot

    @property
    def min(self) -> "Disulfide":
        """Return Disulfide with the minimum energy"""
        sslist = sorted(self.data)
        return sslist[0]

    @property
    def max(self) -> "Disulfide":
        """Return Disulfide with the maximum energy"""
        sslist = sorted(self.data)
        return sslist[-1]

    def minmax_distance(self):
        """Return the Disulfides with min/max Cα distances"""
        sslist = self.data
        if not sslist:
            return None, None
        ssmin = min(sslist, key=lambda ss: ss.ca_distance)
        ssmax = max(sslist, key=lambda ss: ss.ca_distance)
        return ssmin, ssmax

    @property
    def minmax_energy(self):
        """Return the Disulfides with min/max energies"""
        sslist = self.data
        if not sslist:
            return None, None
        sslist = sorted(sslist, key=lambda ss: ss.energy)
        return sslist[0], sslist[-1]

    def get_by_name(self, name):
        """Returns the Disulfide with the given name"""
        for ss in self.data:
            if ss.name == name:
                return ss.copy()
        return None

    def get_chains(self):
        """Return the chain IDs for chains within the Disulfide"""
        res_dict = {"xxx"}
        sslist = self.data
        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            res_dict.update(pchain)
            res_dict.update(dchain)
        res_dict.remove("xxx")
        return res_dict

    def has_chain(self, chain) -> bool:
        """Returns True if given chain contained in Disulfide"""
        chns = {"xxx"}
        chns = self.get_chains()
        return chain in chns

    def by_chain(self, chain: str):
        """Return a DisulfideList from the input chain identifier"""
        reslist = DisulfideList([], chain)
        sslist = self.data
        for ss in sslist:
            pchain = ss.proximal_chain
            dchain = ss.distal_chain
            if pchain == dchain:
                if pchain == chain:
                    reslist.append(ss)
            else:
                print(f"Cross chain SS: {ss.repr_compact}:")
        return reslist

    def nearest_neighbors(self, cutoff: float, *args):
        """
        Find and return neighboring Disulfide objects based on torsion angle similarity.

        This method uses the provided torsion angles to create a model Disulfide object and then
        returns all Disulfide objects in the current list (self) whose torsion angle Euclidean
        distance from the model is less than or equal to the specified cutoff value. The torsion
        angles can be provided either as a single list (or numpy array) of 5 angles or as 5 individual
        float arguments.

        :param cutoff: The maximum allowable Euclidean distance (in degrees) between the torsion angles
                    of the model and another Disulfide for them to be considered neighbors.
        :type cutoff: float
        :param args: Either a single iterable (list or numpy.ndarray) containing 5 torsion angles or 5
                    individual torsion angle float values.
        :type args: list or numpy.ndarray or 5 individual float values
        :return: A DisulfideList containing all Disulfide objects whose torsion distance from the model
                is within the specified cutoff.
        :rtype: DisulfideList
        :raises ValueError: If the number of provided angles is not exactly 5.

        Example:
            >>> import proteusPy as pp
            >>> pdb_list = pp.Load_PDB_SS(verbose=False, subset=True).SSList
            >>> # Using 5 individual angles:
            >>> neighbors = pdb_list.nearest_neighbors(8, -60, -60, -90, -60, -60)
        """

        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 5:
            chi1, chi2, chi3, chi4, chi5 = args[0]
        elif len(args) == 1 and isinstance(args[0], np.ndarray) and len(args[0]) == 5:
            chi1, chi2, chi3, chi4, chi5 = args[0]
        elif len(args) == 5:
            chi1, chi2, chi3, chi4, chi5 = args
        else:
            raise ValueError(
                "You must provide either 5 individual angles or a list of 5 angles."
            )

        modelss = Disulfide("model", torsions=[chi1, chi2, chi3, chi4, chi5])
        res = modelss.torsion_neighbors(self, cutoff)
        resname = f"Neighbors within {cutoff:.2f}° of [{', '.join(f'{angle:.2f}' for angle in modelss.dihedrals)}]"
        res.pdb_id = resname
        return res

    def nearest_neighbors_ss(self, ss: "Disulfide", cutoff: float):
        """Return Disulfides within the torsional cutoff of the input Disulfide"""
        res = ss.torsion_neighbors(self, cutoff)
        resname = f"{ss.name} neighbors within {cutoff}°"
        res.pdb_id = resname
        return res

    def getlist(self):
        """Return a copy of the list"""
        return self.copy()

    def build_ss_from_idlist(self, id_list):
        """Build a DisulfideList from a list of PDB IDs"""
        result = []
        for ss in self.data:
            if ss.pdb_id in id_list:
                result.append(ss.copy())
        return DisulfideList(result, f"Built from {','.join(id_list)}")

    def pprint(self):
        """Pretty print self"""
        sslist = self.data
        for ss in sslist:
            ss.pprint()

    def pprint_all(self):
        """Pretty print full disulfide descriptions"""
        sslist = self.data
        for ss in sslist:
            ss.pprint_all()

    def validate_ss(self, value):
        """Return the Disulfide object if valid"""
        if value is None:
            raise ValueError("The value cannot be None.")

        if not isinstance(value, Disulfide):
            raise TypeError("The value must be an instance of Disulfide.")
        return value

    # Delegate to DisulfideStats
    def build_distance_df(self):
        """Create a dataframe containing distances and energy"""
        return DisulfideStats.build_distance_df(self.data, self.quiet)

    def build_torsion_df(self):
        """Create a dataframe containing torsional parameters"""
        return DisulfideStats.build_torsion_df(self.data, self.quiet)

    def create_deviation_dataframe(self, verbose=False):
        """Create a DataFrame with deviation information"""
        return DisulfideStats.create_deviation_dataframe(self.data, verbose)

    def calculate_torsion_statistics(self):
        """Calculate torsion and distance statistics"""
        return DisulfideStats.calculate_torsion_statistics(self.data)

    def extract_distances(self, distance_type="sg", comparison="less", cutoff=-1):
        """Extract and filter distance values"""
        return DisulfideStats.extract_distances(
            self.data, distance_type, comparison, cutoff
        )

    # Delegate to DisulfideVisualization
    def display(self, style="sb", light="auto", panelsize=512):
        """Display the Disulfide list in the specific rendering style.

        :param style: Rendering style: One of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color
        :param light: If True, light background, if False, dark
        :param panelsize: Size of each panel in pixels"""
        DisulfideVisualization.display_sslist(self, style, light, panelsize)

    def display_overlay(
        self,
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        light="auto",
        winsize=(1024, 1024),
    ):
        """Display all disulfides in the list overlaid in stick mode against
        a common coordinate frame.

        :param screenshot: Save a screenshot
        :param movie: Save a movie
        :param verbose: Verbosity
        :param fname: Filename to save for the movie or screenshot
        :param light: Background color
        :param winsize: Window size tuple (width, height)
        """
        DisulfideVisualization.display_overlay(
            self, screenshot, movie, verbose, fname, light, winsize
        )

    def display_torsion_statistics(
        self,
        display=True,
        save=False,
        fname="ss_torsions.png",
        theme="auto",
    ):
        """Display torsion and distance statistics for a given Disulfide list.

        :param display: Whether to display the plot in the notebook
        :param save: Whether to save the plot as an image file
        :param fname: The name of the image file to save
        :param theme: The theme to use for the plot
        """
        DisulfideVisualization.display_torsion_statistics(
            self, display, save, fname, theme
        )

    # Properties for easy DataFrame access
    @property
    def distance_df(self):
        """Return the distance DataFrame"""
        return self.build_distance_df()

    @property
    def torsion_df(self):
        """Return the torsion DataFrame"""
        return self.build_torsion_df()

    @property
    def torsion_array(self):
        """Return the torsions as an array"""
        return np.array([ss.torsion_array for ss in self.data])

    def filter_by_ca_distance(self, distance: float = -1.0, minimum: float = 2.0):
        """Return a DisulfideList filtered by to between the maxium Ca distance and
        the minimum, which defaults to 2.0A.

        :param distance: Distance in Å
        :param minimum: Distance in Å
        :return: DisulfideList containing disulfides with the given distance.
        """
        reslist = []
        sslist = self.data

        # if distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        reslist = [
            ss
            for ss in sslist
            if ss.ca_distance < distance and ss.ca_distance > minimum
        ]

        return DisulfideList(reslist, f"filtered by distance < {distance:.2f}")

    def filter_by_sg_distance(self, distance: float = -1.0, minimum: float = 1.0):
        """Return a DisulfideList filtered by to between the maxium Sg distance and
        the minimum, which defaults to 1.0A.

        :param distance: Distance in Å
        :param minimum: Distance in Å
        :return: DisulfideList containing disulfides with the given distance.
        """
        reslist = []
        sslist = self.data

        # if distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        reslist = [
            ss
            for ss in sslist
            if ss.sg_distance < distance and ss.sg_distance > minimum
        ]

        return DisulfideList(reslist, f"filtered by Sγ distance < {distance:.2f}")

    def filter_by_distance(
        self, distance_type: str = "ca", distance: float = -1.0, minimum: float = 2.0
    ):
        """
        Return a DisulfideList filtered by the specified distance type (Ca or Sg) between the maximum distance and
        the minimum, which defaults to 2.0A for Ca and 1.0A for Sg.

        :param distance_type: Type of distance to filter by ('ca' or 'sg').
        :param distance: Distance in Å.
        :param minimum: Minimum distance in Å.
        :return: DisulfideList containing disulfides with the given distance.
        """
        reslist = []
        sslist = self.data

        # Set default minimum distance based on distance_type
        if distance_type == "ca":
            default_minimum = 2.0
        elif distance_type == "sg":
            default_minimum = 1.0
        else:
            raise ValueError("Invalid distance_type. Must be 'ca' or 'sg'.")

        # Use the provided minimum distance or the default
        minimum = minimum if minimum != -1.0 else default_minimum

        # If distance is -1.0, return the entire list
        if distance == -1.0:
            return sslist.copy()

        # Filter based on the specified distance type
        if distance_type == "ca":
            reslist = [
                ss
                for ss in sslist
                if ss.ca_distance < distance and ss.ca_distance > minimum
            ]
        elif distance_type == "sg":
            reslist = [
                ss
                for ss in sslist
                if ss.sg_distance < distance and ss.sg_distance > minimum
            ]

        return DisulfideList(
            reslist, f"filtered by {distance_type} distance < {distance:.2f}"
        )

    def plot_distances(
        self, distance_type="ca", cutoff=-1, comparison="less", theme="auto", log=True
    ):
        """Plot the distance values as a histogram.

        :param distance_type: Type of distance to plot ('sg' or 'ca')
        :param cutoff: Cutoff value for the x-axis title
        :param comparison: If 'less', show distances less than cutoff
        :param theme: The plotly theme to use
        :param log: Whether to use a logarithmic scale for the y-axis
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization

        distances = self.extract_distances(distance_type, comparison, cutoff)
        DisulfideVisualization.plot_distances(
            distances,
            distance_type=distance_type,
            cutoff=cutoff,
            comparison=comparison,
            theme=theme,
            log=log,
        )

    def plot_deviation_scatterplots(self, verbose=False, theme="auto"):
        """
        Plot scatter plots for Bondlength_Deviation, Angle_Deviation Ca_Distance and SG_Distance.

        :param verbose: Whether to display the plot in the notebook
        :param theme: Theme to use for the plot ('auto', 'light', or 'dark')
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization

        dev_df = self.create_deviation_dataframe(verbose)
        DisulfideVisualization.plot_deviation_scatterplots(dev_df, theme=theme)

    def plot_deviation_histograms(self, theme="auto", verbose=True):
        """
        Plot histograms for Bondlength_Deviation, Angle_Deviation, and Ca_Distance.

        :param theme: Theme to use for the plot ('auto', 'light', or 'dark')
        :param verbose: Whether to display verbose output
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization
        dev_df = self.create_deviation_dataframe(verbose)
        DisulfideVisualization.plot_deviation_histograms(dev_df, theme=theme, log=True)


# class for the Disulfide bond
class Disulfide:
    r"""
    This class provides a Python object and methods representing a physical disulfide bond
    either extracted from the RCSB protein databank or built using the
    [proteusPy.Turtle3D](turtle3D.html) class. The disulfide bond is an important
    intramolecular stabilizing structural element and is characterized by:

    * Atomic coordinates for the atoms N, Cα, Cβ, C', Sγ for both residues.
    These are stored as both raw atomic coordinates as read from the RCSB file
    and internal local coordinates.
    * The dihedral angles Χ1 - Χ5 for the disulfide bond
    * A name, by default {pdb_id}{prox_resnumb}{prox_chain}_{distal_resnum}{distal_chain}
    * Proximal residue number
    * Distal residue number
    * Approximate bond torsional energy (kcal/mol):

    $$
    E_{kcal/mol} \\approx 2.0 * cos(3.0 * \\chi_{1}) + cos(3.0 * \\chi_{5}) + cos(3.0 * \\chi_{2}) +
    $$
    $$
    cos(3.0 * \chi_{4}) + 3.5 * cos(2.0 * \chi_{3}) + 0.6 * cos(3.0 * \chi_{3}) + 10.1
    $$

    The equation embodies the typical 3-fold rotation barriers associated with single bonds,
    (Χ1, Χ5, Χ2, Χ4) and a high 2-fold barrier for Χ3, resulting from the partial double bond
    character of the S-S bond. This property leads to two major disulfide families, characterized
    by the sign of Χ3. *Left-handed* disulfides have Χ3 < 0° and *right-handed* disulfides have
    Χ3 > 0°. Within this breakdown there are numerous subfamilies, broadly known as the *hook*,
    *spiral* and *staple*. These are under characgterization.

    * Euclidean length of the dihedral angles (degrees) defined as:
    $$\sqrt(\chi_{1}^{2} + \chi_{2}^{2} + \chi_{3}^{2} + \chi_{4}^{2} + \chi_{5}^{2})$$
    * Cα - Cα distance (Å)
    * Cβ - Cβ distance (Å)
    * Sγ - Sγ distance (Å)
    * The previous C' and next N for both the proximal and distal residues. These are needed
    to calculate the backbone dihedral angles Φ and Ψ.
    * Backbone dihedral angles Φ and Ψ, when possible. Not all structures are complete and
    in those cases the atoms needed may be undefined. In this case the Φ and Ψ angles are set
    to -180°.

    The class also provides a rendering capabilities using the excellent [PyVista](https://pyvista.org)
    library, and can display disulfides interactively in a variety of display styles:
    * 'sb' - Split Bonds style - bonds colored by their atom type
    * 'bs' - Ball and Stick style - split bond coloring with small atoms
    * 'pd' - Proximal/Distal style - bonds colored *Red* for proximal residue and *Green* for
    the distal residue.
    * 'cpk' - CPK style rendering, colored by atom type:
        * Carbon   - Grey
        * Nitrogen - Blue
        * Sulfur   - Yellow
        * Oxygen   - Red
        * Hydrogen - White

    Individual renderings can be saved to a file, and animations created.
    """

    def __init__(
        self,
        name: str = "SSBOND",
        proximal: int = -1,
        distal: int = -1,
        proximal_chain: str = "A",
        distal_chain: str = "A",
        pdb_id: str = "1egs",
        quiet: bool = True,
        torsions: list = None,
    ) -> None:
        """
        Initialize the class to defined internal values. If torsions are provided, the
        Disulfide object is built using the torsions and initialized.

        :param name: Disulfide name, by default "SSBOND"
        :param proximal: Proximal residue number, by default -1
        :param distal: Distal residue number, by default -1
        :param proximal_chain: Chain identifier for the proximal residue, by default "A"
        :param distal_chain: Chain identifier for the distal residue, by default "A"
        :param pdb_id: PDB identifier, by default "1egs"
        :param quiet: If True, suppress output, by default True
        :param torsions: List of torsion angles, by default None
        """
        self.name = name
        self.proximal = proximal
        self.distal = distal
        self.energy = _FLOAT_INIT
        self.proximal_chain = proximal_chain
        self.distal_chain = distal_chain
        self.pdb_id = pdb_id
        self.quiet = quiet
        self.proximal_secondary = "Nosecondary"
        self.distal_secondary = "Nosecondary"
        self.ca_distance = _FLOAT_INIT
        self.cb_distance = _FLOAT_INIT
        self.sg_distance = _FLOAT_INIT
        self.torsion_array = np.array(
            (_ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT, _ANG_INIT)
        )
        self.phiprox = _ANG_INIT
        self.psiprox = _ANG_INIT
        self.phidist = _ANG_INIT
        self.psidist = _ANG_INIT

        # global coordinates for the Disulfide, typically as
        # returned from the PDB file

        self.n_prox = ORIGIN
        self.ca_prox = ORIGIN
        self.c_prox = ORIGIN
        self.o_prox = ORIGIN
        self.cb_prox = ORIGIN
        self.sg_prox = ORIGIN
        self.sg_dist = ORIGIN
        self.cb_dist = ORIGIN
        self.ca_dist = ORIGIN
        self.n_dist = ORIGIN
        self.c_dist = ORIGIN
        self.o_dist = ORIGIN

        # set when we can't find previous or next prox or distal
        # C' or N atoms.
        self.missing_atoms = False
        self.modelled = False
        self.resolution = -1.0

        # need these to calculate backbone dihedral angles
        self.c_prev_prox = ORIGIN
        self.n_next_prox = ORIGIN
        self.c_prev_dist = ORIGIN
        self.n_next_dist = ORIGIN

        # local coordinates for the Disulfide, computed using the Turtle3D in
        # Orientation #1. these are generally private.

        self._n_prox = ORIGIN
        self._ca_prox = ORIGIN
        self._c_prox = ORIGIN
        self._o_prox = ORIGIN
        self._cb_prox = ORIGIN
        self._sg_prox = ORIGIN
        self._sg_dist = ORIGIN
        self._cb_dist = ORIGIN
        self._ca_dist = ORIGIN
        self._n_dist = ORIGIN
        self._c_dist = ORIGIN
        self._o_dist = ORIGIN

        # need these to calculate backbone dihedral angles
        self._c_prev_prox = ORIGIN
        self._n_next_prox = ORIGIN
        self._c_prev_dist = ORIGIN
        self._n_next_dist = ORIGIN

        # Dihedral angles for the disulfide bond itself, set to _ANG_INIT
        self.chi1 = _ANG_INIT
        self.chi2 = _ANG_INIT
        self.chi3 = _ANG_INIT
        self.chi4 = _ANG_INIT
        self.chi5 = _ANG_INIT
        self._rho = _ANG_INIT  # new dihedral angle: Nprox - Ca_prox - Ca_dist - N_dist

        self.torsion_length = _FLOAT_INIT

        if torsions is not None and len(torsions) == 5:
            # computes energy, torsion length and rho
            self.dihedrals = torsions
            self.build_yourself()

    # comparison operators, used for sorting. keyed to SS bond energy
    def __lt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy < other.energy
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Disulfide):
            return self.energy <= other.energy
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy > other.energy
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Disulfide):
            return self.energy >= other.energy
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Disulfide):
            return (
                math.isclose(self.torsion_length, other.torsion_length, rel_tol=1e-1)
                and self.proximal == other.proximal
                and self.distal == other.distal
            )
        return False

    def __ne__(self, other):
        if isinstance(other, Disulfide):
            return self.proximal != other.proximal or self.distal != other.distal
        return NotImplemented

    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        s1 = self.repr_ss_info()
        res = f"{s1}>"
        return res

    def _handle_SS_exception(self, message: str):
        """
        This method catches an exception that occurs in the Disulfide
        object (if quiet), or raises it again, this time adding the
        PDB line number to the error message. (private).

        :param message: Error message
        :raises DisulfideConstructionException: Fatal construction exception.

        """
        # message = "%s at line %i." % (message)
        message = f"{message}"

        if self.quiet:
            # just print a warning - some residues/atoms may be missing
            _logger.warning(f"DisulfideConstructionException: {message}\n")
        else:
            # exceptions are fatal - raise again with new message (including line nr)
            raise DisulfideConstructionException(message) from None

    @property
    def binary_class_string(self):
        """
        Return a binary string representation of the disulfide bond class.
        """
        return DisulfideClassManager.class_string_from_dihedral(
            self.chi1, self.chi2, self.chi3, self.chi4, self.chi5, base=2
        )

    @property
    def octant_class_string(self):
        """
        Return the octant string representation of the disulfide bond class.
        """
        return DisulfideClassManager.class_string_from_dihedral(
            self.chi1, self.chi2, self.chi3, self.chi4, self.chi5, base=8
        )

    @property
    def bond_angle_ideality(self):
        """
        Calculate all bond angles for a disulfide bond and compare them to idealized angles.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond angles and idealized bond angles.
        :rtype: float
        """
        return DisulfideStats.bond_angle_ideality(self)

    @property
    def bond_length_ideality(self):
        """
        Calculate bond lengths for a disulfide bond and compare them to idealized lengths.

        :param np.ndarray atom_coordinates: Array containing coordinates of atoms in the order:
            N1, CA1, C1, O1, CB1, SG1, N2, CA2, C2, O2, CB2, SG2
        :return: RMS difference between calculated bond lengths and idealized bond lengths.
        :rtype: float
        """
        return DisulfideStats.bond_length_ideality(self)

    @property
    def internal_coords_array(self):
        """
        Return an array of internal coordinates for the disulfide bond.

        This function collects the coordinates of the backbone atoms involved in the
        disulfide bond and returns them as a numpy array.

        :param self: The instance of the Disulfide class.
        :type self: Disulfide
        :return: A numpy array containing the coordinates of the atoms.
        :rtype: np.ndarray
        """
        coords = []
        coords.append(self._n_prox.get_array())
        coords.append(self._ca_prox.get_array())
        coords.append(self._c_prox.get_array())
        coords.append(self._o_prox.get_array())
        coords.append(self._cb_prox.get_array())
        coords.append(self._sg_prox.get_array())
        coords.append(self._n_dist.get_array())
        coords.append(self._ca_dist.get_array())
        coords.append(self._c_dist.get_array())
        coords.append(self._o_dist.get_array())
        coords.append(self._cb_dist.get_array())
        coords.append(self._sg_dist.get_array())

        return np.array(coords)

    @property
    def coords_array(self):
        """
        Return an array of coordinates for the disulfide bond.

        This function collects the coordinates of backbone atoms involved in the
        disulfide bond and returns them as a numpy array.

        :param self: The instance of the Disulfide class.
        :type self: Disulfide
        :return: A numpy array containing the coordinates of the atoms.
        :rtype: np.ndarray
        """
        coords = []
        coords.append(self.n_prox.get_array())
        coords.append(self.ca_prox.get_array())
        coords.append(self.c_prox.get_array())
        coords.append(self.o_prox.get_array())
        coords.append(self.cb_prox.get_array())
        coords.append(self.sg_prox.get_array())
        coords.append(self.n_dist.get_array())
        coords.append(self.ca_dist.get_array())
        coords.append(self.c_dist.get_array())
        coords.append(self.o_dist.get_array())
        coords.append(self.cb_dist.get_array())
        coords.append(self.sg_dist.get_array())

        return np.array(coords)

    @property
    def dihedrals(self) -> list:
        """
        Return a list containing the dihedral angles for the disulfide.

        """
        return [self.chi1, self.chi2, self.chi3, self.chi4, self.chi5]

    @dihedrals.setter
    def dihedrals(self, dihedrals: list) -> None:
        """
        Sets the disulfide dihedral angles to the inputs specified in the list and
        computes the torsional energy and length of the disulfide bond.

        :param dihedrals: list of dihedral angles.
        """
        self.chi1 = dihedrals[0]
        self.chi2 = dihedrals[1]
        self.chi3 = dihedrals[2]
        self.chi4 = dihedrals[3]
        self.chi5 = dihedrals[4]
        self.torsion_array = np.array(dihedrals)
        self._compute_torsional_energy()
        self._compute_torsion_length()
        self._compute_rho()

    def bounding_box(self) -> np.array:
        """
        Return the bounding box array for the given disulfide.

        :return: np.array
            Array containing the min, max for X, Y, and Z respectively.
            Does not currently take the atom's radius into account.
        """
        coords = self.internal_coords

        xmin, ymin, zmin = coords.min(axis=0)
        xmax, ymax, zmax = coords.max(axis=0)

        res = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])

        return res

    def build_yourself(self) -> None:
        """
        Build a model Disulfide based its internal dihedral state
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinates. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.
        """
        self.build_model(self.chi1, self.chi2, self.chi3, self.chi4, self.chi5)

    def build_model(
        self, chi1: float, chi2: float, chi3: float, chi4: float, chi5: float
    ) -> None:
        """
        Build a model Disulfide based on the input dihedral angles.
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinates. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.

        :param chi1: Chi1 (degrees)
        :param chi2: Chi2 (degrees)
        :param chi3: Chi3 (degrees)
        :param chi4: Chi4 (degrees)
        :param chi5: Chi5 (degrees)

        Example:
        >>> import proteusPy as pp
        >>> modss = pp.Disulfide('model')
        >>> modss.build_model(-60, -60, -90, -60, -60)
        >>> modss.display(style='sb', light="auto")
        """

        self.dihedrals = [chi1, chi2, chi3, chi4, chi5]
        self.proximal = 1
        self.distal = 2

        tmp = Turtle3D("tmp")
        tmp.Orientation = 1

        n = ORIGIN
        ca = ORIGIN
        cb = ORIGIN
        c = ORIGIN

        self.ca_prox = tmp._position
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_prox = n
        self.ca_prox = ca
        self.c_prox = c
        self.cb_prox = cb

        tmp.bbone_to_schain()
        tmp.move(1.53)
        tmp.roll(self.chi1)
        tmp.yaw(112.8)
        self.cb_prox = Vector3D(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi2)
        tmp.yaw(103.8)
        self.sg_prox = Vector3D(tmp._position)

        tmp.move(2.044)
        tmp.roll(self.chi3)
        tmp.yaw(103.8)
        self.sg_dist = Vector3D(tmp._position)

        tmp.move(1.86)
        tmp.roll(self.chi4)
        tmp.yaw(112.8)
        self.cb_dist = Vector3D(tmp._position)

        tmp.move(1.53)
        tmp.roll(self.chi5)
        tmp.pitch(180.0)

        tmp.schain_to_bbone()

        n, ca, cb, c = build_residue(tmp)

        self.n_dist = n
        self.ca_dist = ca
        self.c_dist = c
        self._compute_torsional_energy()
        self._compute_local_coords()
        self._compute_torsion_length()
        self._compute_rho()
        self.ca_distance = distance3d(self.ca_prox, self.ca_dist)
        self.cb_distance = distance3d(self.cb_prox, self.cb_dist)
        self.sg_distance = distance3d(self.sg_prox, self.sg_dist)
        self.torsion_array = np.array([chi1, chi2, chi3, chi4, chi5])
        self.missing_atoms = True
        self.modelled = True

    @property
    def cofmass(self) -> np.array:
        """
        Return the geometric center of mass for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        :return: 3D array for the geometric center of mass
        """

        res = self.internal_coords.mean(axis=0)
        return res

    @property
    def coord_cofmass(self) -> np.array:
        """
        Return the geometric center of mass for the global coordinates of
        the given Disulfide. Missing atoms are not included.

        :return: 3D array for the geometric center of mass
        """

        res = self.coords.mean(axis=0)
        return res

    def copy(self):
        """
        Copy the Disulfide.

        :return: A copy of self.
        """
        return copy.deepcopy(self)

    def _compute_local_coords(self) -> None:
        """
        Compute the internal coordinates for a properly initialized Disulfide Object.

        :param self: SS initialized Disulfide object
        :returns: None, modifies internal state of the input
        """

        turt = Turtle3D("tmp")
        # get the coordinates as np.array for Turtle3D use.
        cpp = self.c_prev_prox.get_array()
        nnp = self.n_next_prox.get_array()

        n = self.n_prox.get_array()
        ca = self.ca_prox.get_array()
        c = self.c_prox.get_array()
        cb = self.cb_prox.get_array()
        o = self.o_prox.get_array()
        sg = self.sg_prox.get_array()

        sg2 = self.sg_dist.get_array()
        cb2 = self.cb_dist.get_array()
        ca2 = self.ca_dist.get_array()
        c2 = self.c_dist.get_array()
        n2 = self.n_dist.get_array()
        o2 = self.o_dist.get_array()

        cpd = self.c_prev_dist.get_array()
        nnd = self.n_next_dist.get_array()

        turt.orient_from_backbone(n, ca, c, cb, ORIENT_SIDECHAIN)

        # internal (local) coordinates, stored as Vector objects
        # to_local returns np.array objects

        self._n_prox = Vector3D(turt.to_local(n))
        self._ca_prox = Vector3D(turt.to_local(ca))
        self._c_prox = Vector3D(turt.to_local(c))
        self._o_prox = Vector3D(turt.to_local(o))
        self._cb_prox = Vector3D(turt.to_local(cb))
        self._sg_prox = Vector3D(turt.to_local(sg))

        self._c_prev_prox = Vector3D(turt.to_local(cpp))
        self._n_next_prox = Vector3D(turt.to_local(nnp))
        self._c_prev_dist = Vector3D(turt.to_local(cpd))
        self._n_next_dist = Vector3D(turt.to_local(nnd))

        self._n_dist = Vector3D(turt.to_local(n2))
        self._ca_dist = Vector3D(turt.to_local(ca2))
        self._c_dist = Vector3D(turt.to_local(c2))
        self._o_dist = Vector3D(turt.to_local(o2))
        self._cb_dist = Vector3D(turt.to_local(cb2))
        self._sg_dist = Vector3D(turt.to_local(sg2))

    def _compute_torsional_energy(self) -> float:
        """
        Compute the approximate torsional energy for the Disulfide's
        conformation and sets its internal state.

        :return: Energy (kcal/mol)
        """
        # @TODO find citation for the ss bond energy calculation

        def torad(deg):
            return np.radians(deg)

        chi1 = self.chi1
        chi2 = self.chi2
        chi3 = self.chi3
        chi4 = self.chi4
        chi5 = self.chi5

        energy = 2.0 * (cos(torad(3.0 * chi1)) + cos(torad(3.0 * chi5)))
        energy += cos(torad(3.0 * chi2)) + cos(torad(3.0 * chi4))
        energy += 3.5 * cos(torad(2.0 * chi3)) + 0.6 * cos(torad(3.0 * chi3)) + 10.1

        self.energy = energy
        return energy

    def display(
        self, single=True, style="sb", light="auto", shadows=False, winsize=WINSIZE
    ) -> None:
        """
        Display the Disulfide bond in the specific rendering style.

        :param single: Display the bond in a single panel in the specific style.
        :param style:  Rendering style: One of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color
        :param light: If True, light background, if False, dark

        Example:
        >>> import proteusPy as pp

        >>> PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
        >>> ss = PDB_SS[0]
        >>> ss.display(style='cpk', light="auto")
        >>> ss.screenshot(style='bs', fname='proteus_logo_sb.png')
        """
        DisulfideVisualization.display_ss(self, single, style, light, shadows, winsize)

        return

    @property
    def TorsionEnergy(self) -> float:
        """
        Return the energy of the Disulfide bond.
        """
        return self._compute_torsional_energy()

    @property
    def TorsionLength(self) -> float:
        """
        Return the energy of the Disulfide bond.
        """
        return self._compute_torsion_length()

    def Distance_neighbors(self, others: DisulfideList, cutoff: float) -> DisulfideList:
        """
        Return list of Disulfides whose RMS atomic distance is within
        the cutoff (Å) in the others list.

        :param others: DisulfideList to search
        :param cutoff: Distance cutoff (Å)
        :return: DisulfideList within the cutoff
        """

        res = [ss.copy() for ss in others if self.Distance_RMS(ss) < cutoff]
        return DisulfideList(res, "neighbors")

    def Distance_RMS(self, other) -> float:
        """
        Calculate the RMS distance between the internal coordinates of self and another Disulfide.
        :param other: Comparison Disulfide
        :return: RMS distance (Å)
        """

        # Get internal coordinates of both objects
        ic1 = self.internal_coords
        ic2 = other.internal_coords

        # Compute the sum of squared differences between corresponding internal coordinates
        totsq = sum(math.dist(p1, p2) ** 2 for p1, p2 in zip(ic1, ic2))

        # Compute the mean of the squared distances
        totsq /= len(ic1)

        # Take the square root of the mean to get the RMS distance
        return math.sqrt(totsq)

    def Torsion_RMS(self, other) -> float:
        """
        Calculate the RMS distance between the dihedral angles of self and another Disulfide.

        :param other: Disulfide object to compare against.
        :type other: Disulfide
        :return: RMS distance in degrees.
        :rtype: float
        :raises ValueError: If the torsion arrays of self and other are not of equal length.
        """
        # Get internal coordinates of both objects
        ic1 = self.torsion_array
        ic2 = other.torsion_array

        # Ensure both torsion arrays have the same length
        if len(ic1) != len(ic2):
            raise ValueError("Torsion arrays must be of the same length.")

        # Compute the total squared difference between corresponding internal coordinates
        total_squared_diff = sum(
            (angle1 - angle2) ** 2 for angle1, angle2 in zip(ic1, ic2)
        )
        mean_squared_diff = total_squared_diff / len(ic1)

        # Return the square root of the mean squared difference as the RMS distance
        return math.sqrt(mean_squared_diff)

    def get_chains(self) -> tuple:
        """
        Return the proximal and distal chain IDs for the Disulfide.

        :return: tuple (proximal, distal) chain IDs
        """
        prox = self.proximal_chain
        dist = self.distal_chain
        return prox, dist

    def get_full_id(self) -> tuple:
        """
        Return the Disulfide full IDs (Used with BIO.PDB)

        :return: Disulfide full IDs
        """
        return (self.proximal, self.distal)

    @property
    def internal_coords(self) -> np.array:
        """
        Return the internal coordinates for the Disulfide.

        :return: Array containing the coordinates, [16][3].
        """
        return self._internal_coords()

    def _internal_coords(self) -> np.array:
        """
        Return the internal coordinates for the Disulfide.
        If there are missing atoms the extra atoms for the proximal
        and distal N and C are set to [0,0,0]. This is needed for the center of
        mass calculations, used when rendering.

        :return: Array containing the coordinates, [16][3].
        """

        # if we don't have the prior and next atoms we initialize those
        # atoms to the origin so as to not effect the center of mass calculations
        if self.missing_atoms:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )
            )
        else:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                    self._c_prev_prox.get_array(),
                    self._n_next_prox.get_array(),
                    self._c_prev_dist.get_array(),
                    self._n_next_dist.get_array(),
                )
            )
        return res_array

    @property
    def coords(self) -> np.array:
        """
        Return the coordinates for the Disulfide as an array.

        :return: Array containing the coordinates, [16][3].
        """
        return self._coords()

    def _coords(self) -> np.array:
        """
        Return the coordinates for the Disulfide as an array.
        If there are missing atoms the extra atoms for the proximal
        and distal N and C are set to [0,0,0]. This is needed for the center of
        mass calculations, used when rendering.

        :return: Array containing the coordinates, [16][3].
        """

        # if we don't have the prior and next atoms we initialize those
        # atoms to the origin so as to not effect the center of mass calculations
        if self.missing_atoms:
            res_array = np.array(
                (
                    self.n_prox.get_array(),
                    self.ca_prox.get_array(),
                    self.c_prox.get_array(),
                    self.o_prox.get_array(),
                    self.cb_prox.get_array(),
                    self.sg_prox.get_array(),
                    self.n_dist.get_array(),
                    self.ca_dist.get_array(),
                    self.c_dist.get_array(),
                    self.o_dist.get_array(),
                    self.cb_dist.get_array(),
                    self.sg_dist.get_array(),
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )
            )
        else:
            res_array = np.array(
                (
                    self.n_prox.get_array(),
                    self.ca_prox.get_array(),
                    self.c_prox.get_array(),
                    self.o_prox.get_array(),
                    self.cb_prox.get_array(),
                    self.sg_prox.get_array(),
                    self.n_dist.get_array(),
                    self.ca_dist.get_array(),
                    self.c_dist.get_array(),
                    self.o_dist.get_array(),
                    self.cb_dist.get_array(),
                    self.sg_dist.get_array(),
                    self.c_prev_prox.get_array(),
                    self.n_next_prox.get_array(),
                    self.c_prev_dist.get_array(),
                    self.n_next_dist.get_array(),
                )
            )
        return res_array

    def internal_coords_res(self, resnumb) -> np.array:
        """
        Return the internal coordinates for the Disulfide. Missing atoms are not included.

        :return: Array containing the coordinates, [12][3].
        """
        return self._internal_coords_res(resnumb)

    def _internal_coords_res(self, resnumb) -> np.array:
        """
        Return the internal coordinates for the internal coordinates of
        the given Disulfide. Missing atoms are not included.

        :param resnumb: Residue number for disulfide
        :raises DisulfideConstructionWarning: Warning raised if the residue number is invalid
        :return: Array containing the internal coordinates for the disulfide
        """
        res_array = np.zeros(shape=(6, 3))

        if resnumb == self.proximal:
            res_array = np.array(
                (
                    self._n_prox.get_array(),
                    self._ca_prox.get_array(),
                    self._c_prox.get_array(),
                    self._o_prox.get_array(),
                    self._cb_prox.get_array(),
                    self._sg_prox.get_array(),
                )
            )
            return res_array

        elif resnumb == self.distal:
            res_array = np.array(
                (
                    self._n_dist.get_array(),
                    self._ca_dist.get_array(),
                    self._c_dist.get_array(),
                    self._o_dist.get_array(),
                    self._cb_dist.get_array(),
                    self._sg_dist.get_array(),
                )
            )
            return res_array
        else:
            mess = f"-> Disulfide._internal_coords(): Invalid argument. \
             Unable to find residue: {resnumb} "
            raise DisulfideConstructionWarning(mess)

    def make_movie(
        self, style="sb", fname="ssbond.mp4", verbose=False, steps=360
    ) -> None:
        """
        Create an animation for ```self``` rotating one revolution about the Y axis,
        in the given ```style```, saving to ```filename```.

        :param style: Rendering style, defaults to 'sb', one of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color

        :param fname: Output filename, defaults to ```ssbond.mp4```
        :param verbose: Verbosity, defaults to False
        :param steps: Number of steps for one complete rotation, defaults to 360.
        """

        DisulfideVisualization.make_movie(self, style, fname, verbose, steps)

    def spin(self, style="sb", verbose=False, steps=360, theme="auto") -> None:
        """
        Spin the object by rotating it one revolution about the Y axis in the given style.

        :param style: Rendering style, defaults to 'sb', one of:
            * 'sb' - split bonds
            * 'bs' - ball and stick
            * 'cpk' - CPK style
            * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
            * 'plain' - boring single color

        :param verbose: Verbosity, defaults to False
        :param steps: Number of steps for one complete rotation, defaults to 360.
        """
        DisulfideVisualization.spin(self, style, verbose, steps, theme)

    def pprint(self) -> None:
        """
        Pretty print general info for the Disulfide
        """
        s1 = self.repr_ss_info()
        s2 = self.repr_ss_ca_dist()
        s2b = self.repr_ss_sg_dist()
        s3 = self.repr_ss_conformation()
        s4 = self.repr_ss_torsion_length()
        res = f"{s1} \n{s3} \n{s2} \n{s2b} \n{s4}>"
        print(res)

    def pprint_all(self) -> None:
        """
        Pretty print all info for the Disulfide
        """
        s1 = self.repr_ss_info() + "\n"
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s4b = self.repr_phipsi()
        s6 = self.repr_ss_ca_dist()
        s6b = self.repr_ss_cb_dist()
        s6c = self.repr_ss_sg_dist()
        s7 = self.repr_ss_torsion_length()
        s8 = self.repr_ss_secondary_structure()

        res = f"{s1} {s2} {s3} {s4}\n {s4b}\n {s6}\n {s6b}\n {s6c}\n {s7}\n {s8}>"

        print(res)

    # repr functions. The class is large, so I split it up into sections
    def repr_ss_info(self) -> str:
        """
        Representation for the Disulfide class
        """
        s1 = f"<Disulfide {self.name}, Source: {self.pdb_id}, Resolution: {self.resolution} Å"
        return s1

    def repr_ss_coords(self) -> str:
        """
        Representation for Disulfide coordinates
        """
        s2 = f"\nProximal Coordinates:\n   N: {self.n_prox}\n   Cα: {self.ca_prox}\n   C: {self.c_prox}\n   O: {self.o_prox}\n   Cβ: {self.cb_prox}\n   Sγ: {self.sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n"
        s3 = f"Distal Coordinates:\n   N: {self.n_dist}\n   Cα: {self.ca_dist}\n   C: {self.c_dist}\n   O: {self.o_dist}\n   Cβ: {self.cb_dist}\n   Sγ: {self.sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n\n"
        stot = f"{s2} {s3}"
        return stot

    def repr_ss_conformation(self) -> str:
        """
        Representation for Disulfide conformation
        """
        s4 = f"Χ1-Χ5: {self.chi1:.2f}°, {self.chi2:.2f}°, {self.chi3:.2f}°, {self.chi4:.2f}° {self.chi5:.2f}°, {self.rho:.2f}°, {self.energy:.2f} kcal/mol"
        stot = f"{s4}"
        return stot

    def repr_ss_local_coords(self) -> str:
        """
        Representation for the Disulfide internal coordinates.
        """
        s2i = f"Proximal Internal Coords:\n   N: {self._n_prox}\n   Cα: {self._ca_prox}\n   C: {self._c_prox}\n   O: {self._o_prox}\n   Cβ: {self._cb_prox}\n   Sγ: {self._sg_prox}\n   Cprev {self.c_prev_prox}\n   Nnext: {self.n_next_prox}\n"
        s3i = f"Distal Internal Coords:\n   N: {self._n_dist}\n   Cα: {self._ca_dist}\n   C: {self._c_dist}\n   O: {self._o_dist}\n   Cβ: {self._cb_dist}\n   Sγ: {self._sg_dist}\n   Cprev {self.c_prev_dist}\n   Nnext: {self.n_next_dist}\n"
        stot = f"{s2i}{s3i}"
        return stot

    def repr_ss_residue_ids(self) -> str:
        """
        Representation for Disulfide chain IDs
        """
        return f"Proximal Residue fullID: <{self.proximal}> Distal Residue fullID: <{self.distal}>"

    def repr_ss_ca_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Cα Distance: {self.ca_distance:.2f} Å"
        return s1

    def repr_ss_cb_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Cβ Distance: {self.cb_distance:.2f} Å"
        return s1

    def repr_ss_sg_dist(self) -> str:
        """
        Representation for Disulfide Ca distance
        """
        s1 = f"Sγ Distance: {self.sg_distance:.2f} Å"
        return s1

    def repr_ss_torsion_length(self) -> str:
        """
        Representation for Disulfide torsion length
        """
        s1 = f"Torsion length: {self.torsion_length:.2f} deg"
        return s1

    def repr_ss_secondary_structure(self) -> str:
        """
        Representation for Disulfide secondary structure
        """
        s1 = f"Proximal secondary: {self.proximal_secondary} Distal secondary: {self.distal_secondary}"
        return s1

    def repr_phipsi(self) -> str:
        """
        Representation for Disulfide phi psi angles
        """
        s1 = f"PhiProx: {self.phiprox:.2f}° PsiProx: {self.psiprox:.2f}°, PhiDist: {self.phidist:.2f}° PsiDist: {self.psidist:.2f}°"
        return s1

    def repr_all(self) -> str:
        """
        Return a string representation for all Disulfide information
        contained in self.
        """

        s1 = self.repr_ss_info() + "\n"
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s4b = self.repr_phipsi()
        s6 = self.repr_ss_ca_dist()
        s8 = self.repr_ss_cb_dist()
        s7 = self.repr_ss_torsion_length()
        s9 = self.repr_ss_secondary_structure()

        res = f"{s1} {s2} {s3} {s4} {s4b} {s6} {s7} {s8} {s9}>"
        return res

    def repr_compact(self) -> str:
        """
        Return a compact representation of the Disulfide object
        :return: string
        """
        return f"{self.repr_ss_info()} {self.repr_ss_conformation()}"

    def repr_conformation(self) -> str:
        """
        Return a string representation of the Disulfide object's conformation.
        :return: string
        """
        return f"{self.repr_ss_conformation()}"

    def repr_coords(self) -> str:
        """
        Return a string representation of the Disulfide object's coordinates.
        :return: string
        """
        return f"{self.repr_ss_coords()}"

    def repr_internal_coords(self) -> str:
        """
        Return a string representation of the Disulfide object's internal coordinaes.
        :return: string
        """
        return f"{self.repr_ss_local_coords()}"

    def repr_chain_ids(self) -> str:
        """
        Return a string representation of the Disulfide object's chain ids.
        :return: string
        """
        return f"{self.repr_ss_residue_ids()}"

    @property
    def rho(self) -> float:
        """
        Return the dihedral angle rho for the Disulfide.
        """
        return self._compute_rho()

    @rho.setter
    def rho(self, value: float):
        """
        Set the dihedral angle rho for the Disulfide.
        """
        self._rho = value

    def _compute_rho(self) -> float:
        """
        Compute the dihedral angle rho for a Disulfide object and
        sets the internal state of the object.
        """

        v1 = self.n_prox - self.ca_prox
        v2 = self.c_prox - self.ca_prox
        n1 = np.cross(v2.get_array(), v1.get_array())

        v4 = self.n_dist - self.ca_dist
        v3 = self.c_dist - self.ca_dist
        n2 = np.cross(v4.get_array(), v3.get_array())
        self._rho = calc_dihedral(
            Vector3D(n1), self.ca_prox, self.ca_dist, Vector3D(n2)
        )
        return self._rho

    def reset(self) -> None:
        """
        Resets the disulfide object to its initial state. All distances,
        angles and positions are reset. The name is unchanged.
        """
        self.__init__(self)

    def same_chains(self) -> bool:
        """
        Function checks if the Disulfide is cross-chain or not.

        Returns
        -------
        bool \n
            True if the proximal and distal residues are on the same chains,
            False otherwise.
        """

        (prox, dist) = self.get_chains()
        return prox == dist

    def screenshot(
        self,
        single=True,
        style="sb",
        fname="ssbond.png",
        verbose=False,
        shadows=False,
        light="Auto",
    ) -> None:
        """
        Create and save a screenshot of the Disulfide in the given style
        and filename

        :param single: Display a single vs panel view, defaults to True
        :param style: Rendering style, one of:
        * 'sb' - split bonds
        * 'bs' - ball and stick
        * 'cpk' - CPK style
        * 'pd' - Proximal/Distal style - Red=proximal, Green=Distal
        * 'plain' - boring single color,
        :param fname: output filename,, defaults to 'ssbond.png'
        :param verbose: Verbosit, defaults to False
        :param shadows: Enable shadows, defaults to False
        """

        DisulfideVisualization.screenshot(
            self, single, style, fname, verbose, shadows, light
        )

    def set_positions(
        self,
        n_prox: Vector3D,
        ca_prox: Vector3D,
        c_prox: Vector3D,
        o_prox: Vector3D,
        cb_prox: Vector3D,
        sg_prox: Vector3D,
        n_dist: Vector3D,
        ca_dist: Vector3D,
        c_dist: Vector3D,
        o_dist: Vector3D,
        cb_dist: Vector3D,
        sg_dist: Vector3D,
        c_prev_prox: Vector3D,
        n_next_prox: Vector3D,
        c_prev_dist: Vector3D,
        n_next_dist: Vector3D,
    ) -> None:
        """
        Set the atomic coordinates for all atoms in the Disulfide object.

        :param n_prox: Proximal N position
        :param ca_prox: Proximal Cα position
        :param c_prox: Proximal C' position
        :param o_prox: Proximal O position
        :param cb_prox: Proximal Cβ position
        :param sg_prox: Proximal Sγ position
        :param n_dist: Distal N position
        :param ca_dist: Distal Cα position
        :param c_dist: Distal C' position
        :param o_dist: Distal O position
        :param cb_dist: Distal Cβ position
        :param sg_dist: Distal Sγ position
        :param c_prev_prox: Proximal previous C'
        :param n_next_prox: Proximal next N
        :param c_prev_dist: Distal previous C'
        :param n_next_dist: Distal next N
        """

        # deep copy
        self.n_prox = n_prox.copy()
        self.ca_prox = ca_prox.copy()
        self.c_prox = c_prox.copy()
        self.o_prox = o_prox.copy()
        self.cb_prox = cb_prox.copy()
        self.sg_prox = sg_prox.copy()
        self.sg_dist = sg_dist.copy()
        self.cb_dist = cb_dist.copy()
        self.ca_dist = ca_dist.copy()
        self.n_dist = n_dist.copy()
        self.c_dist = c_dist.copy()
        self.o_dist = o_dist.copy()

        self.c_prev_prox = c_prev_prox.copy()
        self.n_next_prox = n_next_prox.copy()
        self.c_prev_dist = c_prev_dist.copy()
        self.n_next_dist = n_next_dist.copy()
        self._compute_local_coords()

    def set_name(self, namestr="Disulfide") -> None:
        """
        Set the Disulfide's name.

        :param namestr: Name, by default "Disulfide"
        """
        self.name = namestr

    def set_resnum(self, proximal: int, distal: int) -> None:
        """
        Set the proximal and residue numbers for the Disulfide.

        :param proximal: Proximal residue number
        :param distal: Distal residue number
        """
        self.proximal = proximal
        self.distal = distal

    def _compute_torsion_length(self) -> float:
        """
        Compute the 5D Euclidean length of the Disulfide object. Update the disulfide internal state.

        :return: Torsion length (Degrees)
        """
        # Use numpy array to compute element-wise square
        tors2 = np.square(self.torsion_array)

        # Compute the sum of squares using numpy's sum function
        dist = math.sqrt(np.sum(tors2))

        # Update the internal state
        self.torsion_length = dist

        return dist

    def torsion_distance(self, other) -> float:
        """
        Calculate the 5D Euclidean distance between `self` and another Disulfide
        object. This is used to compare Disulfide Bond torsion angles to
        determine their torsional similarity via a 5-Dimensional Euclidean distance metric.

        :param other: Comparison Disulfide
        :raises ProteusPyWarning: Warning if `other` is not a Disulfide object
        :return: Euclidean distance (Degrees) between `self` and `other`.
        """

        # Check length of torsion arrays
        if len(self.torsion_array) != 5 or len(other.torsion_array) != 5:
            raise ProteusPyWarning(
                "--> Torsion_Distance() requires vectors of length 5!"
            )

        # Convert to numpy arrays
        p1 = np.array(self.torsion_array)
        p2 = np.array(other.torsion_array)

        # Compute the difference and handle angle wrapping
        diff = np.abs(p1 - p2)
        diff = np.where(diff > 180, 360 - diff, diff)

        # Compute the 5D Euclidean distance using numpy's linalg.norm function
        dist = np.linalg.norm(diff)

        return dist

    def torsion_neighbors(self, others: DisulfideList, cutoff: float) -> DisulfideList:
        """
        Return a list of Disulfides within the angular cutoff in the others list.
        This routine is used to find Disulfides having the same torsion length
        within the others list. This is used to find families of Disulfides with
        similar conformations. Assumes self is properly initialized.

        *NB* The routine will not distinguish between +/-
        dihedral angles. *i.e.* [-60, -60, -90, -60, -60] would have the same
        torsion length as [60, 60, 90, 60, 60], two clearly different structures.

        :param others: ```DisulfideList``` to search
        :param cutoff: Dihedral angle degree cutoff
        :return: DisulfideList within the cutoff

        Example:
        In this example we load the disulfide database subset, find the disulfides with
        the lowest and highest energies, and then find the nearest conformational neighbors.
        Finally, we display the neighbors overlaid against a common reference frame.

        >>> import proteusPy as pp
        >>> _theme = pp.set_pyvista_theme("auto")
        >>> PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)

        We point to the complete list to search for lowest and highest energies.
        >>> sslist = PDB_SS.SSList
        >>> ssmin_enrg, ssmax_enrg = PDB_SS.SSList.minmax_energy

        Make an empty list and find the nearest neighbors within 10 degrees avg RMS in
        sidechain dihedral angle space.

        >>> low_energy_neighbors = ssmin_enrg.torsion_neighbors(sslist, 8)

        Display the number found, and then display them overlaid onto their common reference frame.

        >>> tot = low_energy_neighbors.length
        >>> print(f'Neighbors: {tot}')
        Neighbors: 9
        >>> low_energy_neighbors.display_overlay(light="auto")

        """

        res = [ss for ss in others if self.torsion_distance(ss) <= cutoff]
        return DisulfideList(res, "neighbors")

    def translate(self, translation_vector: Vector3D):
        """Translate the Disulfide object by the given vector."""

        self.n_prox += translation_vector
        self.ca_prox += translation_vector
        self.c_prox += translation_vector
        self.o_prox += translation_vector
        self.cb_prox += translation_vector
        self.sg_prox += translation_vector
        self.sg_dist += translation_vector
        self.cb_dist += translation_vector
        self.ca_dist += translation_vector
        self.n_dist += translation_vector
        self.c_dist += translation_vector
        self.o_dist += translation_vector

        self.c_prev_prox += translation_vector
        self.n_next_prox += translation_vector
        self.c_prev_dist += translation_vector
        self.n_next_dist += translation_vector
        self._compute_local_coords()


# Class defination ends


def disulfide_energy_function(x: list) -> float:
    """
    Compute the approximate torsional energy (kcal/mpl) for the input dihedral angles.

    :param x: A list of dihedral angles: [chi1, chi2, chi3, chi4, chi5]
    :return: Energy in kcal/mol

    Example:
    >>> from proteusPy import disulfide_energy_function
    >>> dihed = [-60.0, -60.0, -90.0, -60.0, -90.0]
    >>> res = disulfide_energy_function(dihed)
    >>> float(res)
    2.5999999999999996
    """

    chi1, chi2, chi3, chi4, chi5 = x
    energy = 2.0 * (np.cos(np.deg2rad(3.0 * chi1)) + np.cos(np.deg2rad(3.0 * chi5)))
    energy += np.cos(np.deg2rad(3.0 * chi2)) + np.cos(np.deg2rad(3.0 * chi4))
    energy += (
        3.5 * np.cos(np.deg2rad(2.0 * chi3))
        + 0.6 * np.cos(np.deg2rad(3.0 * chi3))
        + 10.1
    )
    return energy


def minimize_ss_energy(inputSS: Disulfide) -> Disulfide:
    """
    Minimizes the energy of a Disulfide object using the Nelder-Mead optimization method.

    Parameters:
        inputSS (Disulfide): The Disulfide object to be minimized.

    Returns:
        Disulfide: The minimized Disulfide object.

    """

    initial_guess = inputSS.torsion_array
    result = minimize(disulfide_energy_function, initial_guess, method="Nelder-Mead")
    minimum_conformation = result.x
    modelled_min = Disulfide("minimized", minimum_conformation)
    # modelled_min.dihedrals = minimum_conformation
    # modelled_min.build_yourself()
    return modelled_min


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
