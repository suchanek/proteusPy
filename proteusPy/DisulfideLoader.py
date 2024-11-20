"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Author: Eric G. Suchanek, PhD
Last revision: 10/14/2024
"""

# Cα N, Cα, Cβ, C', Sγ Å ° ρ

# pylint: disable=C0301
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=W0612


import copy
import os
import pickle
import sys
import time
import urllib

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
    DATA_DIR,
    LOADER_ALL_URL,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    LOADER_SUBSET_URL,
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
    window using the [display_overlay()](#DisulfideLoader.display_overlay) method. See below for examples.\n

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

        full_path = os.path.join(datadir, _pickleFile)
        if self.verbose and not self.quiet:
            _logger.info(
                f"Reading disulfides from: {full_path}... ",
            )

        try:
            # Check if the file exists before attempting to open it
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")

            with open(full_path, "rb") as f:
                sslist = pickle.load(f)
                old_length = len(sslist)

                filt = DisulfideList(sslist.filter_by_distance(cutoff), "filtered")

                new_length = len(filt)
                if self.verbose and not self.quiet:
                    _logger.info(f"Filtering Ca: old: {old_length}, new: {new_length}")

                old_length = new_length
                filt = filt.filter_by_sg_distance(sg_cutoff)
                new_length = len(filt)
                if self.verbose and not self.quiet:
                    _logger.info(f"Filtering SG: old: {old_length}, new: {new_length}")

                if subset:
                    self.SSList = DisulfideList(filt[:5000], "SUBSET_PDB_SS")
                else:
                    self.SSList = DisulfideList(filt, "ALL_PDB_SS")

                self.TotalDisulfides = len(self.SSList)

        except FileNotFoundError as e:
            _logger.error(f"File not found: {full_path}")
            raise e

        except Exception as e:
            _logger.error(f"An error occurred while loading the file: {full_path}")
            raise e

        self.SSDict = self.create_disulfide_dict()
        self.IDList = list(self.SSDict.keys())

        self.TorsionDF = sslist.torsion_df
        self.TotalDisulfides = len(self.SSList)

        self.tclass = DisulfideClass_Constructor(self, self.verbose)

        if self.verbose and not self.quiet:
            _logger.info("Initialization complete.")

    # overload __getitem__ to handle slicing and indexing, and access by name
    def __getitem__(self, item):
        """
        Implements indexing and slicing to retrieve DisulfideList objects from the
        DisulfideLoader. Supports:

        - Integer indexing to retrieve a single DisulfideList
        - Slicing to retrieve a subset as a DisulfideList
        - Lookup by PDB ID to retrieve all Disulfides for that structure
        - Lookup by full disulfide name

        Raises DisulfideException on invalid indices or names.
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
        Compute and return the average structure resolution for the given list.

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

        Parameters:
        disulfide_list (list): List of disulfide objects.

        Returns:
        dict: Dictionary with pdb_id as keys and lists of indices as values.
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

        if "0" in clsid:
            eightorbin = self.tclass.classdf
        else:
            if base == 8:
                eightorbin = self.tclass.eightclass_df
            elif base == 6:
                eightorbin = self.tclass.sixclass_df

        tot_classes = eightorbin.shape[0]
        class_disulfides = DisulfideList([], clsid, quiet=True)

        if verbose:
            _pbar = tqdm(eightorbin.iterrows(), total=tot_classes, leave=True)
        else:
            _pbar = eightorbin.iterrows()

        for idx, row in _pbar:
            _cls = row["class_id"]
            if _cls == clsid:
                ss_list = row["ss_id"]
                if verbose:
                    pbar = tqdm(ss_list, leave=True)
                else:
                    pbar = ss_list

                for ssid in pbar:
                    class_disulfides.append(self[ssid])

                if verbose:
                    pbar.set_postfix({"Done": ""})
                break

            if verbose:
                _pbar.set_postfix({"Cnt": idx})

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

    def describe(self, quick=True) -> None:
        """
        Display information about the Disulfide database contained in `self`. if `quick` is False
        then display the total RAM used by the object. This takes some time to compute; approximately
        30 seconds on a 2024 MacBook Pro. M3 Max.
        :param quick: If True, don't display the RAM used by the `DisulfideLoader` object.
        :return: None
        """
        vers = self.version
        tot = self.TotalDisulfides
        pdbs = len(self.SSDict)
        ram = 0
        if not quick:
            ram = asizeof.asizeof(self) / (1024 * 1024 * 1024)

        res = self.average_resolution
        cutoff = self.cutoff
        sg_cutoff = self.sg_cutoff
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        ssMin, ssMax = self.SSList.minmax_energy

        print("    =========== RCSB Disulfide Database Summary ==============")
        print(f"       =========== Built: {timestr} ==============")
        print(f"PDB IDs present:                    {pdbs}")
        print(f"Disulfides loaded:                  {tot}")
        print(f"Average structure resolution:       {res:.2f} Å")
        print(f"Lowest Energy Disulfide:            {ssMin.name}")
        print(f"Highest Energy Disulfide:           {ssMax.name}")
        print(f"Cα distance cutoff:                 {cutoff:.2f} Å")
        print(f"Sγ distance cutoff:                 {sg_cutoff:.2f} Å")
        if not quick:
            print(f"Total RAM Used:                     {ram:.2f} GB.")
        print(f"    ================= proteusPy: {vers} =======================")

    def display_overlay(self, pdbid) -> None:
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

        >>> PDB_SS.display_overlay('4yys')

        You can also slice the loader and display as an overly.
        >>> PDB_SS[:8].display_overlay()

        """

        try:
            ssbonds = self[pdbid]
        except KeyError:
            _logger.error("Cannot find key %s in SSBond DB", pdbid)
            return

        ssbonds.display_overlay()
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
            except KeyError:
                mess = f"! Cannot find key {pdbID} in SSBond DB"
                raise DisulfideParseWarning(mess)
        else:
            return copy.deepcopy(self.TorsionDF)

    def list_binary_classes(self):
        """Enumerate the binary classes"""
        for k, v in enumerate(self.tclass.classdict):
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
            if base == 8:
                class_df = self.tclass.filter_eightclass_by_percentage(c)
            elif base == 6:
                class_df = self.tclass.filter_sixclass_by_percentage(c)
            else:
                class_df = self.tclass.filter_eightclass_by_percentage(c)
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

    def plot_binary_to_sixclass_incidence(
        self, theme="light", save=False, savedir=".", verbose=False
    ):
        """
        Plot the incidence of all sextant Disulfide classes for a given binary class.

        :param loader: `proteusPy.DisulfideLoader` object
        """

        if verbose:
            _logger.setLevel("INFO")

        clslist = self.tclass.classdf["class_id"]
        for cls in clslist:
            sixcls = self.tclass.binary_to_six_class(cls)
            df = self.enumerate_class_fromlist(sixcls, base=6)
            self.plot_count_vs_class_df(
                df,
                title=cls,
                theme=theme,
                save=save,
                savedir=savedir,
                base=6,
                verbose=verbose,
            )
        if verbose:
            _logger.info("Graph generation complete.")
            _logger.setLevel("WARNING")

        return

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

        clslist = self.tclass.classdf["class_id"]
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
        Plots a line graph of count vs class ID using Plotly for the given disulfide class. The
        base selects the class type to plot: 2, 6, or 8, for binary, sextant, or octant classes.

        :param df: A pandas DataFrame containing the data to be plotted.
        :param title: A string representing the title of the plot (default is 'title').
        :param theme: A string representing the name of the theme to use. Can be either 'notebook'
        or 'plotly_dark'. Default is 'plotly_dark'.
        :return: None
        """

        _title = f"Binary Class: {title}"
        _labels = {}
        _prefix = "None"
        if base == 8:
            _labels = {"class_id": "Octant Class ID", "count": "Count"}
            _prefix = "Octant"

        elif base == 6:
            _labels = {"class_id": "Sextant Class ID", "count": "Count"}
            _prefix = "Sextant"

        elif base == 2:
            _labels = {"class_id": "Binary Class ID", "count": "Count"}
            _prefix = "Binary"
            df = self.tclass.classdf

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
            fname = os.path.join(savedir, f"{title}_{_prefix}.png")

            if verbose:
                _logger.info("Saving %s plot to %s", title, fname)
            fig.write_image(fname, "png")
        else:
            fig.show()
        return fig

    def plot_count_vs_classid(self, cls=None, title="title", theme="light", base=8):
        """
        Plots a line graph of count vs class ID using Plotly.

        :param df: A pandas DataFrame containing the data to be plotted.
        :param title: A string representing the title of the plot (default is 'title').
        :param theme: A string representing the theme of the plot. Anything other than `light` is in `plotly_dark`.
        :return: None
        """

        _title = f"Binary Class: {title}"

        if base == 8:
            _title = f"Octant Class: {title}"
        elif base == 6:
            _title = f"Sextant Class: {title}"

        df = (
            self.tclass.classdf
            if base == 2
            else self.tclass.sixclass_df if base == 6 else self.tclass.eightclass_df
        )

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

        :param loader: An instance of DisulfideLoader used to load the classes.
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

    def enumerate_sixclass_fromlist(self, sslist) -> pd.DataFrame:
        """
        Enumerates the six-class disulfide bonds from a list of class IDs and
        returns a DataFrame with class IDs and their corresponding counts.

        :param sslist: A list of eight-class disulfide bond class IDs.
        :type sslist: list
        :return: A DataFrame with columns "class_id" and "count" representing the
        class IDs and their corresponding counts.
        :rtype: pd.DataFrame
        """
        x = []
        y = []

        for sixcls in sslist:
            if sixcls is not None:
                _y = self.tclass.sslist_from_classid(sixcls, base=6)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(sixcls)
                    y.append(len(_y))

        sslist_df = pd.DataFrame(columns=["class_id", "count"])
        sslist_df["class_id"] = x
        sslist_df["count"] = y
        return sslist_df

    def enumerate_eightclass_fromlist(self, sslist) -> pd.DataFrame:
        """
        Enumerates the eight-class disulfide bonds from a list of class IDs and
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

    def save(self, savepath=DATA_DIR, subset=False, cutoff=-1.0):
        """
        Save a copy of the fully instantiated Loader to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        :param fname: Filename, defaults to LOADER_FNAME
        :param verbose: Verbosity, defaults to False
        :param cutoff: Distance cutoff used to build the database, -1 means no cutoff.
        """
        self.version = __version__
        self.cutoff = cutoff

        fname = None

        if subset:
            fname = LOADER_SUBSET_FNAME
        else:
            fname = LOADER_FNAME

        _fname = os.path.join(savepath, fname)
        if self.verbose:
            _logger.info("Writing %s...", _fname)

        with open(str(_fname), "wb+") as f:
            pickle.dump(self, f)

        if self.verbose:
            _logger.info("Done saving loader.")


# class ends


def Download_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False):
    """
    Download the databases from my Google Drive.

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to False
    """

    fname = None

    if subset:
        fname = LOADER_SUBSET_FNAME
        url = LOADER_SUBSET_URL
    else:
        fname = LOADER_FNAME
        url = LOADER_ALL_URL

    _fname = os.path.join(loadpath, fname)

    _fname_sub = os.path.join(loadpath, fname)
    _fname_all = os.path.join(loadpath, fname)
    if verbose:
        print("--> DisulfideLoader: Downloading Disulfide Database from Drive...")

    gdown.download(url, str(_fname), quiet=False)
    return


def Download_PDB_SS_GitHub(loadpath=DATA_DIR, verbose=True, subset=False):
    """
    Download the databases from Github. Note: if you change the database these sizes will
    need to be changed!

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to True
    """

    _good1 = 0  # all data
    _good2 = 0  # subset data

    _fname_sub = os.path.join(loadpath, LOADER_SUBSET_FNAME)
    _fname_all = os.path.join(loadpath, LOADER_FNAME)

    _all_length = 340371775
    _subset_length = 9636086

    if verbose:
        print("--> DisulfideLoader: Downloading Disulfide Database from GitHub...")

    _, headers = urllib.request.urlretrieve(
        "https://github.com/suchanek/proteusPy/raw/master/data/PDB_SS_ALL_LOADER.pkl",
        _fname_all,
    )
    num_bytes = headers.get("content-length")
    if num_bytes == _all_length:
        _good1 = 1
    else:
        print(f"--> Read: {num_bytes}, expecting: {_all_length}")

    if subset:
        if verbose:
            print(
                "--> DisulfideLoader: Downloading Disulfide Subset Database from GitHub..."
            )

        _, headers = urllib.request.urlretrieve(
            "https://github.com/suchanek/proteusPy/raw/master/data/PDB_SS_SUBSET_LOADER.pkl",
            _fname_sub,
        )
        num_bytes = headers.get("content-length")
        if num_bytes == _subset_length:
            _good2 = 1
        else:
            print(f"--> Read: {num_bytes}, expecting: {_subset_length}")
    return _good1 + _good2


def Load_PDB_SS(
    loadpath=DATA_DIR, verbose=False, subset=False, cutoff=8.0, force=False
) -> DisulfideLoader:
    """
    Load the fully instantiated Disulfide database from the specified file. Use the
    defaults unless you are building the database by hand. *This is the function
    used to load the built database.*

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to False
    :param subset: If True, load the subset DB, otherwise load the full database
    """
    # normally the .pkl files are local, EXCEPT for the first run from a newly-installed proteusPy
    # distribution. In that case we need to download the files for all disulfides and the subset
    # from the GitHub.

    _good1 = False  # all data
    _good2 = False  # subset data

    _fname_sub = os.path.join(loadpath, LOADER_SUBSET_FNAME)
    _fname_all = os.path.join(loadpath, LOADER_FNAME)

    if subset:
        if not os.path.exists(_fname_sub) or force is True:
            loader = Bootstrap_PDB_SS(
                loadpath=loadpath,
                verbose=verbose,
                subset=True,
                force=force,
                cutoff=cutoff,
            )
            loader.save(savepath=loadpath, subset=True, cutoff=cutoff)
            return loader
        else:
            if verbose:
                print(f"-> load_PDB_SS(): Reading {_fname_sub}... ")

            with open(_fname_sub, "rb") as f:
                subloader = pickle.load(f)
            return subloader
    else:
        if not os.path.exists(_fname_all) or force is True:
            loader = Bootstrap_PDB_SS(
                loadpath=loadpath,
                verbose=verbose,
                subset=False,
                force=force,
                cutoff=cutoff,
            )
            loader.save(savepath=loadpath, subset=False, cutoff=cutoff)
            if verbose:
                print(f"-> load_PDB_SS(): Done Saving {_fname_all}... ")
            return loader
        else:
            if verbose:
                print(f"-> load_PDB_SS(): Reading {_fname_all}... ")

            with open(_fname_all, "rb") as f:
                loader = pickle.load(f)

            if verbose:
                print(f"-> load_PDB_SS(): Done Reading {_fname_all}... ")

    return loader


def Bootstrap_PDB_SS(
    loadpath=DATA_DIR, cutoff=8.0, verbose=False, subset=False, force=False
):
    """
    Download and instantiate the disulfide databases from Google Drive.

    This function downloads the disulfide master SS list from Google Drive if it doesn't
    already exist in the specified load path or if the force flag is set to True.
    It then loads the disulfide data from the downloaded file and initializes a
    DisulfideLoader instance.

    :param loadpath: Path from which to load the data, defaults to DATA_DIR
    :type loadpath: str
    :param cutoff: Cutoff value for disulfide loading, defaults to 8.0
    :type cutoff: float
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

    _fname = os.path.join(loadpath, fname)

    if not os.path.exists(_fname) or force is True:
        if verbose:
            _logger.info("Downloading Disulfide Database from Drive...")
        gdown.download(url, str(_fname), quiet=False)

    full_path = os.path.join(loadpath, _fname)
    if verbose:
        _logger.info("Building loader from: %s...", full_path)

    loader = DisulfideLoader(
        datadir=DATA_DIR, subset=subset, verbose=verbose, cutoff=cutoff
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
