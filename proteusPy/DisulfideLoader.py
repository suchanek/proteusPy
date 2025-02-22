"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Last revision: 2025-02-21 16:33:56 -egs-
"""

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

# pylint: disable=C0301
# pylint: disable=C0302
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=W0612

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

import copy
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import gdown
import pandas as pd
import plotly_express as px
from pympler import asizeof

from proteusPy import __version__
from proteusPy.DisulfideBase import Disulfide, DisulfideList
from proteusPy.DisulfideClassManager import DisulfideClassManager
from proteusPy.DisulfideExceptions import DisulfideParseWarning
from proteusPy.DisulfideStats import DisulfideStats
from proteusPy.DisulfideVisualization import DisulfideVisualization
from proteusPy.logger_config import create_logger
from proteusPy.ProteusGlobals import (
    CA_CUTOFF,
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SG_CUTOFF,
    SS_LIST_URL,
    SS_PICKLE_FILE,
)

_logger = create_logger(__name__)


try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__  # type: ignore
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


@dataclass
class DisulfideLoader:
    """
    This class represents the disulfide database itself and is its primary means of accession.
    The entirety of the RCSB disulfide database is stored within the class via a
    proteusPy.DisulfideList, a ```Pandas``` .csv file, and a ```dict``` of
    indices mapping the PDB IDs into their respective list of disulfides. The datastructures allow
    simple, direct and flexible access to the disulfide structures contained herein.
    This makes it possible to access the disulfides by array index, PDB structure ID, disulfide
    name and class ID.

    The class also provides methods for plotting distance and angle deviations
    as well as torsion statistics for the disulfides in the database.

    The class can also render Disulfides overlaid on a common coordinate system to a pyVista
    window using the [display_overlay()](#DisulfideLoader.display_overlay) method. See below for examples.

    Important note: For typical usage one will access the database via the `Load_PDB_SS()` function.
    The difference is that the latter function loads the compressed database from its single
    source. The `DisulfideLoader` class is used to build the Disulifde database with a
    specific cutoff, or for saving the database to a file.

    :param verbose: Flag to control output verbosity
    :type verbose: bool
    :param datadir: Directory containing data files
    :type datadir: str
    :param picklefile: Name of the pickle file containing disulfide data
    :type picklefile: str
    :param quiet: Flag to suppress warnings
    :type quiet: bool
    :param subset: Flag to load only a subset of data
    :type subset: bool
    :param cutoff: Distance cutoff for filtering disulfides
    :type cutoff: float
    :param sg_cutoff: SG distance cutoff for filtering disulfides
    :type sg_cutoff: float
    """

    # Instance attributes
    SSList: DisulfideList = field(
        default_factory=lambda: DisulfideList([], "ALL_PDB_SS")
    )
    SSDict: Dict = field(default_factory=dict)
    TorsionDF: pd.DataFrame = field(default_factory=pd.DataFrame)
    TotalDisulfides: int = field(default=0)
    IDList: List = field(default_factory=list)
    quiet: bool = field(default=False)
    tclass: Optional[DisulfideClassManager] = field(default=None)
    cutoff: float = field(default=-1.0)
    sg_cutoff: float = field(default=-1.0)
    verbose: bool = field(default=False)
    percentile: float = field(default=-1.0)
    timestamp: float = field(default_factory=time.time)
    version: str = field(default=__version__)

    # Initialization parameters
    datadir: str = field(default=DATA_DIR)
    picklefile: str = field(default=SS_PICKLE_FILE)
    subset: bool = field(default=False)
    cutoff: float = field(default=-1.0)
    sg_cutoff: float = field(default=-1.0)
    verbose: bool = field(default=False)
    percentile: float = field(default=-1.0)

    def __post_init__(self) -> None:
        """
        Initialize the DisulfideLoader after dataclass initialization.
        This method handles loading and processing of the disulfide data.
        """

        cutoffs = {}
        old_length = new_length = 0
        full_path = Path(self.datadir) / self.picklefile

        if self.verbose and not self.quiet:
            _logger.info(
                f"Reading disulfides from: {full_path}... ",
            )

        try:
            # Check if the file exists before attempting to open it
            if not full_path.exists():
                fname = SS_PICKLE_FILE
                url = SS_LIST_URL

                _fname = Path(DATA_DIR) / fname

                if not _fname.exists():
                    if self.verbose:
                        _logger.info(
                            "Master SS list unavailable. Downloading Disulfide Database from Drive..."
                        )
                    gdown.download(url, str(_fname), quiet=False)

            with open(full_path, "rb") as f:
                sslist = pickle.load(f)

                if self.percentile > 0.0:
                    if self.percentile > 100.0:
                        raise ValueError("Percentile must be between 0 and 100.")

                    cutoffs = DisulfideStats.calculate_cutoff_from_percentile(
                        sslist, percentile=self.percentile, verbose=self.verbose
                    )

                    ca_cutoff = cutoffs["ca_cutoff_percentile"]
                    sg_cutoff = cutoffs["sg_cutoff_percentile"]
                    self.cutoff = ca_cutoff
                    self.sg_cutoff = sg_cutoff

                    if self.verbose:
                        _logger.info(
                            f"Using percentile cutoffs: {ca_cutoff:.2f}, {sg_cutoff:.2f}"
                        )

                old_length = len(sslist)
                filt = sslist.filter_by_distance(
                    distance=self.cutoff, distance_type="ca"
                )
                filt = DisulfideList(
                    filt,
                    f"filtered by Ca cutoff:{self.cutoff:.2f}, Sg cutoff: {self.sg_cutoff:.2f}",
                )

                new_length = len(filt)

                if self.verbose:
                    _logger.info(
                        "Filtered with Cα cutoff %.2f: old: %d, new: %d",
                        self.cutoff,
                        old_length,
                        new_length,
                    )

                old_length = new_length
                filt = filt.filter_by_distance(
                    distance=self.sg_cutoff, distance_type="sg"
                )
                new_length = len(filt)

                if self.verbose:
                    _logger.info(
                        "Filtered with Sγ: cutoff %.2f: old: %d, new: %d",
                        self.sg_cutoff,
                        old_length,
                        new_length,
                    )
                if self.subset:
                    self.SSList = DisulfideList(filt[:5000], "SUBSET_PDB_SS")
                else:
                    self.SSList = DisulfideList(filt, "ALL_PDB_SS")

                self.SSDict = self.create_disulfide_dict()
                self.IDList = list(self.SSDict.keys())

                self.TorsionDF = self.SSList.torsion_df
                self.TotalDisulfides = len(self.SSList)
                self.tclass = DisulfideClassManager(self, self.verbose)

            if self.verbose:
                _logger.info("Loader initialization complete.")
                self.describe()

        except FileNotFoundError as e:
            _logger.error("File not found: %s", full_path)
            raise e

        except Exception as e:
            _logger.error("An error occurred while loading the file: %s", full_path)
            raise e

    # overload __getitem__ to handle slicing and indexing, and access by name or classid
    def __getitem__(self, item):
        """
        Implements indexing and slicing to retrieve DisulfideList objects from the
        DisulfideLoader. Supports:

        - Integer indexing to retrieve a single DisulfideList
        - Slicing to retrieve a subset as a DisulfideList
        - Lookup by PDB ID to retrieve all Disulfides for that structure
        - Lookup by full disulfide name
        - Lookup by classid in the format 11111b or 11111o. The last char is the class type.
        - Lookup by classid in the format 11111. The base is 8 by default.

        :param index: The index or key to retrieve the DisulfideList.
        :type index: int, slice, str
        :return: A DisulfideList object or a subset of it.
        :rtype: DisulfideList
        :raises DisulfideException: If the index or name is invalid.
        """

        res = DisulfideList([], "none")
        ind_list = []

        if isinstance(item, slice):
            indices = range(*item.indices(len(self.SSList)))
            ind_list = list(indices)
            name = f"pdb_slice[{ind_list[0]}:{ind_list[-1]+1}]"
            resolution = self.SSList[0].resolution
            sublist = [self.SSList[i] for i in indices]
            return DisulfideList(sublist, name, resolution)

        if isinstance(item, int):
            if item < 0 or item >= self.TotalDisulfides:
                _logger.error(
                    "DisulfideLoader(): Index %d out of range 0-%d",
                    item,
                    self.TotalDisulfides - 1,
                )
                return res

            res = self.SSList[item]
            return res

        # if the item is a string, it could be a PDB ID or a full disulfide name
        # or a classid in the format 11111b or 11111o. the last char is the class type

        if isinstance(item, str) and len(item) == 6 or len(item) == 5:  # classid
            res = self.extract_class(item, verbose=self.verbose)
            return res

        # PDB_SS['4yys'] return a list of SS
        try:
            indices = self.SSDict[item]
            if indices:
                res = DisulfideList([], item)
                sslist = self.SSList
                for ind in indices:
                    res.append(sslist[ind])
            else:
                # try to find the full disulfide name
                res = self.SSList.get_by_name(item)  # full disulfide name

        except KeyError as e:
            res = self.SSList.get_by_name(item)  # full disulfide name

        if not res:
            _logger.error("DisulfideLoader(): Cannot find key %s in SSBond DB", item)
        return res

    def __setitem__(self, index, item):
        self.SSList[index] = self._validate_ss(item)

    def _validate_ss(self, value):
        if isinstance(value, Disulfide):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")

    @property
    def average_resolution(self) -> float:
        """
        Return the average structure resolution for the given list.
        Result is cached since resolution values don't change after loading.

        :return: Average resolution (A)
        """
        sslist = self.SSList
        valid_resolutions = [
            ss.resolution
            for ss in sslist
            if ss.resolution is not None and ss.resolution != -1.0
        ]

        if not valid_resolutions:
            return -1.0

        return sum(valid_resolutions) / len(valid_resolutions)

    @staticmethod
    def binary_to_class(binary_class: str, base=8) -> str:
        """
        Convert a binary class string to an octant class string.

        :param binary_class: The binary class string to convert.
        :param base: The base class to use, 2 or 8.
        :return: The octant class string.
        """
        return DisulfideClassManager.binary_to_class(binary_class, base)

    def build_ss_from_idlist(self, idlist) -> DisulfideList:
        """
        Return a DisulfideList of Disulfides for a given list of PDBIDs

        :param idlist: List of PDBIDs, e.g. ['4yys', '2q7q']
        :return: DisulfideList
        """
        res = DisulfideList([], "RCSB_list")
        for pdbid, sslist in self.SSDict.items():
            if pdbid in idlist:
                for ssid in sslist:
                    res.append(self.SSList[ssid])
        return res

    def class_indices_from_tors_df(self, class_string, base=8) -> pd.Index:
        """
        Return the row indices of the torsion dataframe that match the class string.

        This method is used internally to find the indices of rows in the torsion dataframe
        that match the specified class string based on the given base.

        :param class_string: The class string to match in the torsion dataframe.
        :type class_string: str
        :param base: The base class to use for matching, either 2 or 8. Defaults to 8.
        :type base: int
        :return: The row indices of the torsion dataframe that match the class string.
        :rtype: pd.Index
        :raises ValueError: If the base is not 2 or 8.
        """
        tors_df = self.TorsionDF
        match base:
            case 8:
                column = "octant_class_string"
            case 2:
                column = "binary_class_string"
            case _:
                raise ValueError(f"Base must be 2 or 8, not {base}")

        return tors_df[tors_df[column] == class_string].index

    def copy(self):
        """
        Return a copy of self.

        :return: Copy of self
        """
        return copy.deepcopy(self)

    def create_disulfide_dict(self):
        """
        Create a dictionary from a list of disulfide objects where the key is the pdb_id
        and the value is a list of indices of the disulfide objects in the list.

        :param disulfide_list: List of disulfide objects.
        :type disulfide_list: list
        :return: Dictionary with pdb_id as keys and lists of indices as values.
        :rtype: dict
        """
        disulfide_list = self.SSList

        disulfide_dict = {}
        for index, disulfide in enumerate(disulfide_list):
            if disulfide.pdb_id not in disulfide_dict:
                disulfide_dict[disulfide.pdb_id] = []
            disulfide_dict[disulfide.pdb_id].append(index)
        return disulfide_dict

    def get_class_df(self, base=8) -> pd.DataFrame:
        """
        Return the class incidence dataframe for the input base.
        Result is cached since class distributions don't change after loading.

        :param base: The base class to use, 2 or 8.
        :return: pd.DataFrame
        """
        return self.tclass.get_class_df(base)

    def extract_class(self, clsid: str, verbose: bool = False) -> DisulfideList:
        """
        Return the list of disulfides corresponding to the input `clsid`.

        :param clsid: The class name to extract.
        :param verbose: If True, display progress bars, by default False
        :return: The list of disulfide bonds from the class.
        """

        cls = clsid[:5]

        try:
            ss_ids = self.tclass[clsid]

        except KeyError:
            _logger.error("Cannot find key %s in SSBond DB", clsid)
            return DisulfideList([], cls, quiet=True)

        tot_ss = len(ss_ids)
        class_disulfides = DisulfideList([], cls, quiet=True)

        _pbar = (
            tqdm(range(tot_ss), total=tot_ss, leave=True) if verbose else range(tot_ss)
        )

        for idx in _pbar:
            ssid = ss_ids[idx]
            class_disulfides.append(self[ssid])

        return class_disulfides

    def getlist(self) -> DisulfideList:
        """
        Return the list of Disulfides contained in the class.

        :return: DisulfideList
        :rtype: DisulfideList
        """
        return copy.deepcopy(self.SSList)

    def get_by_name(self, name) -> Disulfide:
        """
        Return the Disulfide with the given name from the list.
        Result is cached since disulfide data doesn't change after loading.
        """
        for ss in self.SSList.data:
            if ss.name == name:
                return ss  # or ss.copy() !!!
        return None

    def describe(self, memusg=False) -> None:
        """
        Display information about the Disulfide database contained in `self`. if `quick` is False
        then display the total RAM used by the object. This takes some time to compute; approximately
        30 seconds on a 2024 MacBook Pro. M3 Max.

        :param memusg: If True, don't display the RAM used by the `DisulfideLoader` object.
        :return: None
        """
        vers = self.version
        tot = self.TotalDisulfides
        pdbs = len(self.SSDict)
        ram = 0
        if memusg:
            ram = asizeof.asizeof(self) / (1024 * 1024 * 1024)

        res = self.average_resolution
        cutoff = self.cutoff
        sg_cutoff = self.sg_cutoff
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        ssMin, ssMax = self.SSList.minmax_energy

        print("    =========== RCSB Disulfide Database Summary ============")
        print(f"       =========== Built: {timestr} ===========")
        print(f"PDB IDs present:                 {pdbs}")
        print(f"Disulfides loaded:               {tot}")
        print(f"Average structure resolution:    {res:.2f} Å")
        print(f"Lowest Energy Disulfide:         {ssMin.name}")
        print(f"Highest Energy Disulfide:        {ssMax.name}")
        print(f"Cα distance cutoff:              {cutoff:.2f} Å")
        print(f"Sγ distance cutoff:              {sg_cutoff:.2f} Å")
        if memusg:
            print(f"Total RAM Used:                     {ram:.2f} GB.")
        print(f"               ===== proteusPy: {vers} =====")

    def display_overlay(self, pdbid, verbose=False) -> None:
        """
        Display all disulfides for a given PDB ID overlaid in stick mode against
        a common coordinate frame. This allows us to see all of the disulfides
        at one time in a single view. Colors vary smoothy between bonds.

        :param self: DisulfideLoader object initialized with the database.
        :param pdbid: the PDB id string, e.g. 4yys
        :return: None

        Example:
        >>> import proteusPy as pp

        Instantiate the Loader with the SS database subset.

        >>> PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)

        Display the Disulfides from the PDB ID ```4yys```, overlaid onto
        a common reference (the proximal disulfides).

        >>> PDB_SS.display_overlay('4yys', verbose=False)

        You can also slice the loader and display as an overly.
        >>> PDB_SS[:8].display_overlay(verbose=False)

        """

        try:
            ssbonds = self[pdbid]
        except KeyError:
            _logger.error("Cannot find key %s in SSBond DB", pdbid)
            return

        ssbonds.display_overlay(verbose=verbose)
        return

    def getTorsions(self, pdbID=None) -> pd.DataFrame:
        """
        Return the torsions, distances and energies defined by Torsion_DF_cols

        :param pdbID: pdbID, defaults to None, meaning return entire dataset.
        :type pdbID: str, optional used to extract for a specific PDB structure. If not specified
            then return the entire dataset.
        :raises DisulfideParseWarning: Raised if not found
        :return: Torsions Dataframe
        :rtype: pd.DataFrame

        Example:
        >>> import proteusPy as pp
        >>> PDB_SS = pp.Load_PDB_SS(verbose=False, subset=True)
        >>> Tor_DF = PDB_SS.getTorsions()
        """
        res_df = pd.DataFrame()

        if pdbID:
            try:
                res = self.SSDict[pdbID]
                sel = self.TorsionDF["source"] == pdbID
                res_df = self.TorsionDF[sel]
                return res_df.copy()
            except KeyError as e:
                mess = f"Cannot find key {pdbID} in SSBond DB"
                _logger.error(mess)
                raise DisulfideParseWarning(mess) from e
        else:
            return copy.deepcopy(self.TorsionDF)

    def list_binary_classes(self):
        """Enumerate the binary classes"""
        for k, v in enumerate(self.tclass.binaryclass_dict):
            print(f"Class: |{k}|, |{v}|")

    def plot_classes_vs_cutoff(
        self, cutoff: float, steps: int = 50, base=8, theme="auto", verbose=False
    ) -> None:
        """
        Plot the total percentage and number of members for each octant class against the cutoff value.

        :param cutoff: Percent cutoff value for filtering the classes.
        :param steps: Number of steps to take in the cutoff.
        :param base: The base class to use, 6 or 8.
        :param theme: The theme to use for the plot ('auto', 'light', or 'dark'), defaults to 'auto'.
        :return: None
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization

        DisulfideVisualization.plot_classes_vs_cutoff(
            self.tclass, cutoff, steps, base, theme, verbose
        )

    def plot_binary_to_eightclass_incidence(
        self,
        theme="light",
        save=False,
        savedir=".",
        verbose=False,
    ):
        """Plot the incidence of all octant Disulfide classes for a given binary class.

        :param tclass: DisulfideClassManager instance
        :param theme: The theme to use for the plot
        :param save: Whether to save the plots
        :param savedir: Directory to save plots to
        :param verbose: Whether to display verbose output
        """

        DisulfideVisualization.plot_binary_to_eightclass_incidence(
            self.tclass, theme, save, savedir, verbose
        )

    def plot_count_vs_class_df(
        self,
        class_string,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
    ):
        """
        Plot a line graph of count vs class ID using Plotly for the given disulfide class. The
        base selects the class type to plot: 2, 6, or 8, for binary, sextant, or octant classes.

        :param class_string: The binary class string to be plotted.
        :param title: A string representing the title of the plot (default is 'title').
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization
        class_list = self.tclass.binary_to_class(class_string, base)
        df = self.enumerate_class_fromlist(class_list, base=base)
        DisulfideVisualization.plot_count_vs_class_df(
            df, title, theme, save, savedir, base, verbose, log
        )

    def plot_count_vs_class_df_sampled(
        self,
        class_string,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
        sample_size=1000,
    ):
        """
        Plot a line graph of count vs class ID using Plotly for the given disulfide class with sampling.

        :param df: DataFrame containing class data
        :param title: Title for the plot
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        :param sample_size: Number of items to sample
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization
        class_list = self.tclass.binary_to_class(class_string, base)
        df = self.enumerate_class_fromlist(class_list, base=base)

        DisulfideVisualization.plot_count_vs_class_df_sampled(
            df, title, theme, save, savedir, base, verbose, log, sample_size
        )

    def plot_count_vs_class_df_paginated(
        self,
        class_string,
        title="title",
        theme="auto",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
        log=True,
        page_size=200,
    ):
        """
        Plot a line graph of count vs class ID using Plotly for the given disulfide class with pagination.

        :param df: DataFrame containing class data
        :param title: Title for the plot
        :param theme: Theme to use for the plot
        :param save: Whether to save the plot
        :param savedir: Directory to save the plot to
        :param base: Base for class IDs (2 or 8)
        :param verbose: Whether to display verbose output
        :param log: Whether to use log scale for y-axis
        :param page_size: Number of items per page
        """
        class_list = self.tclass.binary_to_class(class_string, base)
        df = self.enumerate_class_fromlist(class_list, base=base)

        DisulfideVisualization.plot_count_vs_class_df_paginated(
            df, title, theme, save, savedir, base, verbose, log, page_size
        )

    def plot_count_vs_classid(self, cls=None, theme="auto", base=8, log=True):
        """
        Plot a line graph of count vs class ID using Plotly.

        :param cls: Specific class to plot (optional)
        :param theme: Theme to use for the plot
        :param base: Base for class IDs (2 or 8)
        :param log: Whether to use log scale for y-axis
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization

        DisulfideVisualization.plot_count_vs_classid(self.tclass, cls, theme, base, log)

    def enumerate_class_fromlist(self, sslist, base=8):
        """
        Enumerate the classes from a list of class IDs and return a DataFrame with class IDs and their corresponding counts.
        Results are cached for improved performance on repeated calls.

        :param sslist: A list of class IDs to enumerate.
        :param base: The base value for the enumeration, by default 8.
        :return: A DataFrame with columns "class_id" and "count" representing the class IDs and their corresponding counts.
        """
        x = []
        y = []

        for cls in sslist:
            if cls is not None:
                _y = self.tclass.sslist_from_classid(cls, base=base)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(cls)
                    y.append(len(_y))

        sslist_df = pd.DataFrame(columns=["class_id", "count"])
        sslist_df["class_id"] = x
        sslist_df["count"] = y
        return sslist_df

    def save(
        self,
        savepath=DATA_DIR,
        subset=False,
        cutoff=-1.0,
        sg_cutoff=-1.0,
        verbose=False,
    ):
        """
        Save a copy of the fully instantiated Loader to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        :param fname: Filename, defaults to LOADER_FNAME
        :param verbose: Verbosity, defaults to False
        :param cutoff: Ca-Ca Distance cutoff used to build the database, -1 means no cutoff.
        :param sg_cutoff: Sg-Sg Distance cutoff used to build the database, -1 means no cutoff.
        """
        self.version = __version__
        self.cutoff = cutoff
        self.sg_cutoff = sg_cutoff

        fname = None

        if subset:
            fname = LOADER_SUBSET_FNAME
        else:
            fname = LOADER_FNAME

        _fname = Path(savepath) / fname

        if verbose:
            _logger.info("Writing Disulfide Loader to: %s...", _fname)

        with open(str(_fname), "wb+") as f:
            pickle.dump(self, f)

        if verbose:
            _logger.info("Done saving loader.")

    def plot_disulfides_vs_pdbid(self, cutoff=1):
        """
        Plots the number of disulfides versus pdbid.

        :param cutoff: The minimum number of disulfides a PDB ID must have to be included in the plot.
        :type cutoff: int
        :return: A tuple containing the list of PDB IDs and the corresponding number of disulfides.
        :rtype: tuple
        """
        pdbids = []
        num_disulfides = []

        for pdbid, disulfides in self.SSDict.items():
            if len(disulfides) > cutoff:
                pdbids.append(pdbid)
                num_disulfides.append(len(disulfides))

        # Create a DataFrame
        df = pd.DataFrame({"PDB ID": pdbids, "Number of Disulfides": num_disulfides})
        fig = px.bar(
            df,
            x="PDB ID",
            y="Number of Disulfides",
            title=f"Disulfides vs PDB ID with cutoff: {cutoff}, {len(pdbids)} PDB IDs",
        )
        fig.update_layout(
            xaxis_title="PDB ID",
            yaxis_title="Number of Disulfides",
            xaxis_tickangle=-90,
        )
        fig.show()

        return pdbids, num_disulfides

    def plot_distances(
        self, distance_type="ca", cutoff=-1, comparison="less", theme="auto", log=True
    ):
        """
        Plot the distances for the disulfides in the loader.

        :param distance_type: The type of distance to plot ('ca' for Cα-Cα distance, 'sg' for Sγ-Sγ distance)
        :param cutoff: The cutoff value for the distance, defaults to -1 (no cutoff)
        :param comparison: if 'less' then plot distances less than the cutoff, if 'greater' then plot distances greater than the cutoff
        :param theme: The theme to use for the plot ('auto', 'light', or 'dark')
        :param log: Whether to use a log scale for the y-axis
        """
        # from proteusPy.DisulfideVisualization import DisulfideVisualization

        # sslist = self.SSList
        # distances = sslist.extract_distances(distance_type, comparison, cutoff)

        self.SSList.plot_distances(
            distance_type=distance_type,
            cutoff=cutoff,
            comparison=comparison,
            theme=theme,
            log=log,
        )

    def plot_deviation_scatterplots(self, verbose=False, theme="auto"):
        """
        Plot scatter plots for Bondlength_Deviation, Angle_Deviation Ca_Distance
        and SG_Distance.

        :param verbose: Whether to display the plot in the notebook. Default is False.
        :type verbose: bool
        :param theme: One of 'Auto', 'Light', or 'Dark'. Default is 'Auto'.
        :type light: str
        :return: None
        """
        self.SSList.plot_deviation_scatterplots(verbose=verbose, theme=theme)

    def plot_deviation_histograms(self, theme="auto", verbose=True):
        """
        Plot histograms for Bondlength_Deviation, Angle_Deviation, and Ca_Distance.
        """
        self.SSList.plot_deviation_histograms(theme=theme, verbose=verbose)

    def plot_classes(
        self, class_string, base=8, theme="auto", log=False, page_size=200
    ):
        """
        Plot the distribution of disulfides for the given class string.

        :param class_string: The class string to plot.
        :type class_string: str
        :param base: The base of the class string. Default is 8.
        :type base: int
        :param theme: The theme to use for the plot ('auto', 'light', or 'dark'), defaults to 'auto'.
        :type theme: str
        :return: None
        """
        classlist = self.tclass.binary_to_class(class_string, base)
        df = self.enumerate_class_fromlist(classlist, base=base)
        DisulfideVisualization.plot_count_vs_class_df_paginated(
            df, title=class_string, theme=theme, base=base, log=log, page_size=page_size
        )

        return

    def sslist_from_class(self, class_string, base=8, cutoff=0.0) -> DisulfideList:
        """
        Return a DisulfideList containing Disulfides with the given class_string.

        :param class_string: The class string to search for.
        :param base: The base of the class string. Default is 8.
        :param cutoff: The % cutoff value for the class. Default is 0.0.
        :return: DisulfideList containing Disulfides with the given class_string.
        """
        sslist_name = f"{class_string}_{base}_{cutoff:.2f}"
        sslist = DisulfideList([], sslist_name)

        indices = self.class_indices_from_tors_df(class_string, base=base)

        for i in indices:
            sslist.append(self[i])

        return sslist

    def display_torsion_statistics(
        self,
        display=True,
        save=False,
        fname="ss_torsions.png",
        theme="auto",
    ):
        """
        Display torsion and distance statistics for all Disulfides in the loader.

        :param display: Whether to display the plot in the notebook. Default is True.
        :type display: bool
        :param save: Whether to save the plot as an image file. Default is False.
        :type save: bool
        :param fname: The name of the image file to save. Default is 'ss_torsions.png'.
        :type fname: str
        :param theme: One of 'Auto', 'Light', or 'Dark'. Default is 'Auto'.
        :type theme: str
        :return: None
        """
        self.SSList.display_torsion_statistics(
            display=display,
            save=save,
            fname=fname,
            theme=theme,
        )

    def classes_vs_cutoff(self, cutoff, base=8):
        """
        Return number of members for the octant class for a given cutoff value.

        :param cutoff: Percent cutoff value for filtering the classes.
        :return: None
        """

        class_df = self.tclass.filter_class_by_percentage(cutoff, base=base)
        return class_df.shape[0]


# class ends


def Load_PDB_SS(
    loadpath=DATA_DIR,
    verbose=False,
    subset=False,
    cutoff=CA_CUTOFF,
    sg_cutoff=SG_CUTOFF,
    force=False,
) -> DisulfideLoader:
    """
    Load the fully instantiated Disulfide database from the specified file. This function
    will load the pre-built database if available, or bootstrap a new loader by downloading
    the data from Google Drive if needed. Use the provided parameters to control the loading
    behavior, filtering cutoffs, and verbosity.

    :param loadpath: Path from which to load the database; defaults to DATA_DIR.
    :type loadpath: str
    :param verbose: If True, enables verbose logging; defaults to False.
    :type verbose: bool
    :param subset: If True, loads the subset database; otherwise loads the full database.
    :type subset: bool
    :param cutoff: Cα distance cutoff used to filter disulfides; defaults to CA_CUTOFF.
    :type cutoff: float
    :param sg_cutoff: Sγ distance cutoff used to filter disulfides; defaults to SG_CUTOFF.
    :type sg_cutoff: float
    :param force: If True, forces re-loading from Google Drive even if the file exists; defaults to False.
    :type force: bool
    :param percentile: Percentile (0-100) to compute cutoffs dynamically; if set to -1.0, the percentile method is not used.
    :type percentile: float
    :return: An instance of DisulfideLoader containing the loaded disulfide database.
    :rtype: DisulfideLoader

    Example:
        >>> from proteusPy import Load_PDB_SS, create_logger
        >>> import logging
        >>> _logger = create_logger("testing")
        >>> _logger.setLevel(logging.WARNING)
        >>> loader = Load_PDB_SS(verbose=False, subset=True)
        >>> print(loader[0])
        <Disulfide 6dmb_203A_226A, Source: 6dmb, Resolution: 3.0 Å>
    """

    # normally the .pkl files are local, EXCEPT for the first run from a newly-installed proteusPy
    # distribution. In that case we need to download the files for all disulfides and the subset
    # from my Google Drive. This is a one-time operation.

    _fname_sub = Path(loadpath) / LOADER_SUBSET_FNAME
    _fname_all = Path(loadpath) / LOADER_FNAME
    _fpath = _fname_sub if subset else _fname_all

    if not _fpath.exists() or force is True:
        if verbose:
            _logger.info(f"Bootstrapping new loader: {str(_fpath)}... ")

        loader = Bootstrap_PDB_SS(
            loadpath=loadpath,
            verbose=verbose,
            subset=subset,
            force=force,
            cutoff=cutoff,
            sg_cutoff=sg_cutoff,
        )
        loader.save(
            savepath=loadpath,
            subset=subset,
            cutoff=cutoff,
            sg_cutoff=sg_cutoff,
        )
        return loader

    if verbose:
        _logger.info("Reading disulfides from: %s...", _fpath)

    with open(_fpath, "rb") as f:
        loader = pickle.load(f)
    if verbose:
        _logger.info("Done reading disulfides from: %s...", _fpath)
        loader.describe()

    return loader


def Bootstrap_PDB_SS(
    loadpath=DATA_DIR,
    cutoff=-1.0,
    sg_cutoff=-1.0,
    verbose=True,
    subset=False,
    force=False,
    fake=False,
):
    """
    Download and instantiate the disulfide databases from Google Drive.

    This function downloads the disulfide master SS list from Google Drive if it doesn't
    already exist in the specified load path or if the force flag is set to True.
    It then loads the disulfide data from the downloaded file and initializes a
    DisulfideLoader instance.

    :param loadpath: Path from which to load the data, defaults to DATA_DIR
    :type loadpath: str
    :param cutoff: Cutoff value for disulfide loading, defaults to -1.0 (no filtering)
    :type cutoff: float
    :param sg_cutoff: Cutoff value for disulfide loading, defaults to -1.0 (no filtering)
    :type sg_cutoff: float
    :param verbose: Flag to enable verbose logging, defaults to False
    :type verbose: bool
    :param subset: Flag to indicate whether to load a subset of the data, defaults to False
    :type subset: bool
    :param force: Flag to force download even if the file exists, defaults to False
    :type force: bool
    :return: An instance of DisulfideLoader initialized with the loaded data
    :rtype: DisulfideLoader
    """

    fname = SS_PICKLE_FILE
    url = SS_LIST_URL

    # _fname = Path(loadpath) / fname
    full_path = Path(loadpath) / fname

    if not full_path.exists() or force is True:
        if verbose:
            _logger.warning("Can't find %s. Downloading from Drive...", full_path)

        if not fake:
            gdown.download(url, str(full_path), quiet=False)
        else:
            if verbose:
                _logger.warning("Fake download: %s", full_path)
                return None
    if verbose:
        _logger.info(
            "Building loader from: %s with cutoffs %f, %f...",
            full_path,
            cutoff,
            sg_cutoff,
        )

    loader = DisulfideLoader(
        datadir=DATA_DIR,
        subset=subset,
        verbose=verbose,
        cutoff=cutoff,
        sg_cutoff=sg_cutoff,
    )

    if loader.TotalDisulfides == 0:
        _logger.error("No disulfides loaded!")
        return None

    if verbose:
        _logger.info("Done building loader.")

    return loader


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
