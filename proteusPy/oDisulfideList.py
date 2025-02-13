"""
This module provides the implementation and interface for the DisulfideList object,
used extensively by Disulfide class.

Author: Eric G. Suchanek, PhD
Last revision: 2025-02-13 00:41:44 -egs-
"""

from collections import UserList
from itertools import combinations

import numpy as np

from proteusPy.DisulfideStats import DisulfideStats
from proteusPy.DisulfideVisualization import DisulfideVisualization
from proteusPy.logger_config import create_logger

_logger = create_logger(__name__)

# pylint: disable=C0103 # snake case
# pylint: disable=C0301 # line too long
# pylint: disable=C0302 # too many lines in module
# pylint: disable=C0405 # import outside toplevel
# pylint: disable=W0212 # access to protected member
# pylint: disable=W0613 # unused argument


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
        """
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.data)))
            ind_list = list(indices)
            first_ind = ind_list[0]
            last_ind = ind_list[-1]
            name = (
                self.data[first_ind].pdb_id
                + f"_slice[{first_ind}:{last_ind+1}]_{self.data[last_ind].pdb_id}"
            )
            sublist = [self.data[i] for i in indices]
            return DisulfideList(sublist, name)
        return UserList.__getitem__(self, item)

    def __setitem__(self, index, item):
        self.data[index] = self.validate_ss(item)

    def append(self, item):
        """Append the list with item"""
        self.data.append(self.validate_ss(item))

    def extend(self, other):
        """Extend the Disulfide list with other"""
        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(self.validate_ss(item) for item in other)

    def insert(self, index, item):
        """Insert a Disulfide into the list at the specified index"""
        self.data.insert(index, self.validate_ss(item))

    @property
    def length(self):
        """Return the length of the list"""
        return len(self.data)

    @property
    def id(self):
        """PDB ID of the list"""
        return self.pdb_id

    @id.setter
    def id(self, value):
        """Set the DisulfideList ID"""
        self.pdb_id = value

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
        """Average conformation [x1, x2, x3, x4, x5]"""
        sslist = self.data
        return np.mean([ss.torsion_array for ss in sslist], axis=0)

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
        """Return all Disulfides within the given angle cutoff"""
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 5:
            chi1, chi2, chi3, chi4, chi5 = args[0]
        elif len(args) == 5:
            chi1, chi2, chi3, chi4, chi5 = args
        else:
            raise ValueError(
                "You must provide either 5 individual angles or a list of 5 angles."
            )

        sslist = self.data
        # Import here to avoid circular dependency
        from proteusPy.Disulfide import Disulfide

        modelss = Disulfide("model", torsions=[chi1, chi2, chi3, chi4, chi5])
        res = modelss.torsion_neighbors(sslist, cutoff)
        resname = f"Neighbors within {cutoff:.2f}° of [{', '.join(f'{angle:.2f}' for angle in modelss.dihedrals)}]"
        res.pdb_id = resname
        return res

    def nearest_neighbors_ss(self, ss, cutoff: float):
        """Return Disulfides within the torsional cutoff of the input Disulfide"""
        sslist = self.data
        res = ss.torsion_neighbors(sslist, cutoff)
        resname = f"{ss.name} neighbors within {cutoff}°"
        res.pdb_id = resname
        return res

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
        # Import here to avoid circular dependency
        from proteusPy.Disulfide import Disulfide

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
        """Display the Disulfide list in the specific rendering style"""
        DisulfideVisualization.display(self.data, style, light, panelsize)

    def display_overlay(
        self,
        screenshot=False,
        movie=False,
        verbose=False,
        fname="ss_overlay.png",
        light="auto",
        winsize=(1024, 1024),
    ):
        """Display all disulfides overlaid in stick mode"""
        DisulfideVisualization.display_overlay(
            self.data, screenshot, movie, verbose, fname, light, winsize
        )

    def display_torsion_statistics(
        self,
        display=True,
        save=False,
        fname="ss_torsions.png",
        theme="auto",
    ):
        """Display torsion and distance statistics"""
        DisulfideVisualization.display_torsion_statistics(
            self.data, display, save, fname, theme
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

    def filter_by_distance(self, distance: float = -1.0, minimum: float = 2.0):
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
