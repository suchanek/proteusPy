"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Last revision: 2025-01-07 22:04:58 -egs-
"""

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

# pylint: disable=C0301
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=W0612

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

import copy
import pickle
import time
from pathlib import Path

import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly_express as px
from pympler import asizeof

from proteusPy import __version__
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideClass_Constructor import DisulfideClass_Constructor
from proteusPy.DisulfideExceptions import DisulfideParseWarning
from proteusPy.DisulfideList import DisulfideList
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


class DisulfideLoader:
    """
    This class represents the disulfide database itself and is its primary means of accession.
    The entirety of the RCSB disulfide database is stored within the class via a
    proteusPy.DisulfideList, a ```Pandas``` .csv file, and a ```dict``` of
    indices mapping the PDB IDs into their respective list of disulfides. The datastructures allow
    simple, direct and flexible access to the disulfide structures contained herein.
    This makes it possible to access the disulfides by array index, PDB structure ID or disulfide
    name.

    The class can also render Disulfides overlaid on a common coordinate system to a pyVista
    window using the [display_overlay()](#DisulfideLoader.display_overlay) method. See below for examples.

    Important note: For typical usage one will access the database via the `Load_PDB_SS()` function.
    The difference is that the latter function loads the compressed database from its single
    source. The `DisulfideLoader` class is used to build the Disulifde database with a
    specific cutoff, or for saving the database to a file.

    *Developer's Notes:*
    The .pkl files needed to instantiate this class and save it into its final .pkl file are
    defined in the proteusPy.data class and should not be changed.
    """

    def __init__(
        self,
        verbose: bool = False,
        datadir: str = DATA_DIR,  # the package installation data directory
        picklefile: str = SS_PICKLE_FILE,  # PDB_all_ss.pkl by default
        quiet: bool = True,
        subset: bool = False,
        cutoff: float = -1.0,
        sg_cutoff: float = -1.0,
    ) -> None:
        """
        Initializing the class initiates loading either the entire Disulfide dataset,
        or the 'subset', which consists of the first 5000 disulfides. The subset
        is useful for testing and debugging since it doesn't require nearly as much
        memory or time. The name for the subset file is hard-coded. One can pass a
        different data directory and file names for the pickle files. These different
        directories are normally established with the proteusPy.Extract_Disulfides
        function.

        """

        self.SSList = DisulfideList([], "ALL_PDB_SS")
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []
        self._quiet = quiet
        self.tclass = None  # disulfideClass_constructor to manage classes
        self.cutoff = cutoff  # distance cutoff used to bulid the database
        self.sg_cutoff = sg_cutoff  # distance cutoff used to bulid the database
        self.verbose = verbose
        self.timestamp = time.time()
        self.version = __version__

        _pickleFile = picklefile
        old_length = new_length = 0

        full_path = Path(datadir) / _pickleFile
        if self.verbose and not self.quiet:
            _logger.info(
                f"Reading disulfides from: {full_path}... ",
            )

        try:
            # Check if the file exists before attempting to open it
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")

            with open(full_path, "rb") as f:
                sslist = pickle.load(f)
                old_length = len(sslist)

                filt = DisulfideList(sslist.filter_by_distance(cutoff), "filtered")

                new_length = len(filt)
                if self.verbose:
                    _logger.info(
                        "Filtering with Cα cutoff %f: old: %d, new: %d",
                        cutoff,
                        old_length,
                        new_length,
                    )

                old_length = new_length
                filt = filt.filter_by_sg_distance(sg_cutoff)
                new_length = len(filt)

                if self.verbose:
                    _logger.info(
                        "Filtering Sγ: cutoff %f: old: %d, new: %d",
                        sg_cutoff,
                        old_length,
                        new_length,
                    )
                if subset:
                    self.SSList = DisulfideList(filt[:5000], "SUBSET_PDB_SS")
                else:
                    self.SSList = DisulfideList(filt, "ALL_PDB_SS")

                self.TotalDisulfides = len(self.SSList)

                self.SSDict = self.create_disulfide_dict()
                self.IDList = list(self.SSDict.keys())

                self.TorsionDF = self.SSList.torsion_df
                self.TotalDisulfides = len(self.SSList)
                self.tclass = DisulfideClass_Constructor(self, self.verbose)

            if self.verbose:
                _logger.info("Loader initialization complete.")

        except FileNotFoundError as e:
            _logger.error("File not found: %s", full_path)
            raise e

        except Exception as e:
            _logger.error("An error occurred while loading the file: %s", full_path)
            raise e

    # overload __getitem__ to handle slicing and indexing, and access by name
    def __getitem__(self, item):
        """
        Implements indexing and slicing to retrieve DisulfideList objects from the
        DisulfideLoader. Supports:
        
        - Integer indexing to retrieve a single DisulfideList
        - Slicing to retrieve a subset as a DisulfideList
        - Lookup by PDB ID to retrieve all Disulfides for that structure
        - Lookup by full disulfide name
        
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
            else:
                return self.SSList[item]

        try:
            # PDB_SS['4yys'] return a list of SS
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
        if isinstance(value, (Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")

    @property
    def average_resolution(self) -> float:
        """
        Return the average structure resolution for the given list.

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

    def extract_class(self, clsid, base=8, verbose=False) -> DisulfideList:
        """
        Return the list of disulfides corresponding to the input `clsid`.

        :param clsid: The class name to extract.
        :param verbose: If True, display progress bars, by default False
        :return: The list of disulfide bonds from the class.
        """

        eightorbin = None

        if base == 8:
            eightorbin = self.tclass.eightclass_dict
        elif base == 2:
            eightorbin = self.tclass.binaryclass_dict
        else:
            raise ValueError("Invalid base value.")

        ss_ids = eightorbin[clsid]

        tot_ss = len(eightorbin)
        class_disulfides = DisulfideList([], clsid, quiet=True)

        if verbose:
            _pbar = tqdm(range(tot_ss), total=tot_ss, leave=True)
        else:
            _pbar = range(tot_ss)

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
        print(f"      ============== proteusPy: {vers} ===================")

    def display_overlay(self, pdbid, verbose=False) -> None:
        """
        Display all disulfides for a given PDB ID overlaid in stick mode against
        a common coordinate frame. This allows us to see all of the disulfides
        at one time in a single view. Colors vary smoothy between bonds.

        :param self: DisulfideLoader object initialized with the database.
        :param pdbid: the PDB id string, e.g. 4yys
        :return: None

        Example:
        >>> from proteusPy import Disulfide, Load_PDB_SS, DisulfideList

        Instantiate the Loader with the SS database subset.

        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)

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
        Return the torsions, distances and energies defined by Disulfide.Torsion_DF_cols

        :param pdbID: pdbID, defaults to None, meaning return entire dataset.
        :type pdbID: str, optional used to extract for a specific PDB structure. If not specified
            then return the entire dataset.
        :raises DisulfideParseWarning: Raised if not found
        :return: Torsions Dataframe
        :rtype: pd.DataFrame

        Example:
        >>> from proteusPy import Load_PDB_SS
        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
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

    @property
    def quiet(self) -> bool:
        """
        The loader quiet state

        :return: quiet parameter
        :rtype: bool
        """
        return self._quiet

    @quiet.setter
    def quiet(self, perm: bool) -> None:
        """
        Sets the quiet attribute for the loader. This silences many of the BIO.PDB warnings.

        :param perm: True or False
        :type perm: bool
        """
        self._quiet = perm

    def plot_classes_vs_cutoff(self, cutoff, steps, base=8) -> None:
        """
        Plot the total percentage and number of members for each octant class against the cutoff value.

        :param cutoff: Percent cutoff value for filtering the classes.
        :param steps: Number of steps to take in the cutoff.
        :param base: The base class to use, 6 or 8.
        :return: None
        """

        _cutoff = np.linspace(0, cutoff, steps)
        tot_list = []
        members_list = []

        for c in _cutoff:
            class_df = self.tclass.filter_class_by_percentage(base, c)
            tot = class_df["percentage"].sum()
            tot_list.append(tot)
            members_list.append(class_df.shape[0])
            print(
                f"Cutoff: {c:5.3} accounts for {tot:7.2f}% and is {class_df.shape[0]:5} members long."
            )

        _, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(_cutoff, tot_list, label="Total percentage", color="blue")
        ax2.plot(_cutoff, members_list, label="Number of members", color="red")

        ax1.set_xlabel("Cutoff")
        ax1.set_ylabel("Total percentage", color="blue")
        ax2.set_ylabel("Number of members", color="red")

        plt.show()

    def plot_binary_to_eightclass_incidence(
        self,
        theme="light",
        save=False,
        savedir=".",
        verbose=False,
    ):
        """
        Plot the incidence of all sextant Disulfide classes for a given binary class.

        :param loader: `proteusPy.DisulfideLoader` object
        """

        if verbose:
            _logger.setLevel("INFO")

        clslist = self.tclass.binaryclass_df["class_id"]
        for cls in clslist:
            eightcls = self.tclass.binary_to_eight_class(cls)
            df = self.enumerate_class_fromlist(eightcls, base=8)
            self.plot_count_vs_class_df(
                df,
                title=cls,
                theme=theme,
                save=save,
                savedir=savedir,
                base=8,
                verbose=verbose,
            )
        if verbose:
            _logger.info("Graph generation complete.")
            _logger.setLevel("WARNING")
        return

    def plot_count_vs_class_df(
        self,
        df,
        title="title",
        theme="light",
        save=False,
        savedir=".",
        base=8,
        verbose=False,
    ):
        """
        Plot a line graph of count vs class ID using Plotly for the given disulfide class. The
        base selects the class type to plot: 2, 6, or 8, for binary, sextant, or octant classes.

        :param df: A pandas DataFrame containing the data to be plotted.
        :param title: A string representing the title of the plot (default is 'title').
        :param theme: A string representing the name of the theme to use. Can be either 'notebook'
        or 'plotly_dark'. Default is 'plotly_dark'.
        :param save: A boolean flag indicating whether to save the plot to a file. Default is False.
        :param savedir: A string representing the directory to save the plot to. Default is '.'.
        :param base: An integer representing the base value for the enumeration. Default is 8.
        :param verbose: A boolean flag indicating whether to display verbose output. Default is False.
        :raises ValueError: If an invalid base value is provided, (2 or 8s).
        :return: None
        """

        _title = f"Binary Class: {title}"
        _labels = {}
        _prefix = "None"
        if base == 8:
            _labels = {"class_id": "Octant Class ID", "count": "Count"}
            _prefix = "Octant"

        elif base == 2:
            _labels = {"class_id": "Binary Class ID", "count": "Count"}
            _prefix = "Binary"
            df = self.tclass.binaryclass_df
        else:
            raise ValueError("Invalid base. Must be 2 or 8.")

        fig = px.line(
            df,
            x="class_id",
            y="count",
            title=f"{_title}",
            labels=_labels,
        )

        if theme == "light":
            fig.update_layout(template="plotly_white")
        else:
            fig.update_layout(template="plotly_dark")

        fig.update_layout(
            showlegend=True,
            title_x=0.5,
            title_font=dict(size=20),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )
        fig.update_layout(autosize=True)

        if save:
            fname = Path(savedir) / f"{title}_{_prefix}.png"

            if verbose:
                _logger.info("Saving %s plot to %s", title, fname)
            fig.write_image(fname, "png")
        else:
            fig.show()
        return fig

    def plot_count_vs_classid(self, cls=None, title="title", theme="light", base=8):
        """
        Plot a line graph of count vs class ID using Plotly.

        :param df: A pandas DataFrame containing the data to be plotted.
        :param title: A string representing the title of the plot (default is 'title').
        :param theme: A string representing the theme of the plot. Anything other than `light` is in `plotly_dark`.
        :return: None
        """

        _title = None

        if base == 8:
            _title = f"Octant Class: {title}"
        elif base == 2:
            _title = f"Binary Class: {title}"
        else:
            raise ValueError("Invalid base. Must be 2 or 8")

        df = self.tclass.binaryclass_df if base == 2 else self.tclass.eightclass_df

        if cls is None:
            fig = px.line(df, x="class_id", y="count", title=_title)
        else:
            subset = df[df["class_id"] == cls]
            fig = px.line(subset, x="class_id", y="count", title=_title)

        fig.update_layout(
            xaxis_title="Class ID",
            yaxis_title="Count",
            showlegend=True,
            title_x=0.5,
        )
        fig.layout.autosize = True

        if theme == "light":
            fig.update_layout(template="plotly_white")
        else:
            fig.update_layout(template="plotly_dark")

        fig.update_layout(autosize=True)
        return fig

    def enumerate_class_fromlist(self, sslist, base=8):
        """
        Enumerate the classes from a list of class IDs and return a DataFrame with class IDs and their corresponding counts.

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

    def enumerate_eightclass_fromlist(self, sslist) -> pd.DataFrame:
        """
        Enumerate the eight-class disulfide bonds from a list of class IDs and
        returns a DataFrame with class IDs and their corresponding counts.

        :param sslist: A list of eight-class disulfide bond class IDs.
        :type sslist: list
        :return: A DataFrame with columns "class_id" and "count" representing the
        class IDs and their corresponding counts.
        :rtype: pd.DataFrame
        """
        x = []
        y = []

        for eightcls in sslist:
            if eightcls is not None:
                _y = self.tclass.sslist_from_classid(eightcls, base=8)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(eightcls)
                    y.append(len(_y))

        sslist_df = pd.DataFrame(columns=["class_id", "count"])
        sslist_df["class_id"] = x
        sslist_df["count"] = y
        return sslist_df

    def save(self, savepath=DATA_DIR, subset=False, cutoff=-1.0, sg_cutoff=-1.0):
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
        if self.verbose:
            _logger.info("Writing %s...", _fname)

        with open(str(_fname), "wb+") as f:
            pickle.dump(self, f)

        if self.verbose:
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

        plt.figure(figsize=(12, 6))
        plt.bar(pdbids, num_disulfides, color="skyblue")
        plt.xlabel("PDB ID")
        plt.ylabel("Number of Disulfides")
        plt.title(f"Number of Disulfides vs PDB ID with cutoff: {cutoff}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        return pdbids, num_disulfides


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
    Load the fully instantiated Disulfide database from the specified file. Use the
    defaults unless you are building the database by hand. *This is the function
    used to load the built database.*

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :type loadpath: str
    :param verbose: Verbosity, defaults to False
    :type verbose: bool
    :param subset: If True, load the subset DB, otherwise load the full database
    :type subset: bool
    :return: The loaded Disulfide database
    :rtype: DisulfideList

    Example:
    >>> from proteusPy import Load_PDB_SS, create_logger
    >>> import logging
    >>> _logger = create_logger("testing")
    >>> _logger.setLevel(logging.WARNING)
    >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
    >>> PDB_SS[0]
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

    return loader


def Bootstrap_PDB_SS(
    loadpath=DATA_DIR,
    cutoff=-1.0,
    sg_cutoff=-1.0,
    verbose=False,
    subset=False,
    force=False,
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

    _fname = Path(loadpath) / fname

    if not _fname.exists() or force is True:
        if verbose:
            _logger.info("Downloading Disulfide Database from Drive...")
        gdown.download(url, str(_fname), quiet=False)

    full_path = Path(loadpath) / _fname
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
