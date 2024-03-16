"""
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Author: Eric G. Suchanek, PhD
Last revision: 2/9/2024
"""

import copy
import pickle
import sys
import time

import pandas as pd

import proteusPy
from proteusPy.atoms import *
from proteusPy.data import *
from proteusPy.data import (
    DATA_DIR,
    LOADER_FNAME,
    LOADER_SUBSET_FNAME,
    SS_DICT_PICKLE_FILE,
    SS_PICKLE_FILE,
    SS_TORSIONS_FILE,
)
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideClass_Constructor import DisulfideClass_Constructor
from proteusPy.DisulfideExceptions import *
from proteusPy.DisulfideList import DisulfideList
from proteusPy.ProteusGlobals import MODEL_DIR, PDB_DIR, REPO_DATA_DIR

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

# Now use tqdm as normal, depending on your environment


class DisulfideLoader:
    """
    This class represents the disulfide database itself and is its primary means of accession.
    The entirety of the RCSB disulfide database is stored within the class via a
    proteusPy.DisulfideList.DisulfideList, a ```Pandas``` .csv file, and a ```dict``` of
    indices mapping the PDB IDs into their respective list of disulfides. The datastructures allow
    simple, direct and flexible access to the disulfide structures contained herein.
    This makes it possible to access the disulfides by array index, PDB structure ID or disulfide name.

    The class can also render Disulfides overlaid on a common coordinate system to a pyVista window using the
    [display_overlay()](#DisulfideLoader.display_overlay) method. See below for examples.\n

    Important note: For typical usage one will access the database via the `Load_PDB_SS()` function.
    The difference is that the latter function loads the compressed database from its single
    source. the `Load_PDB_SS()` function will load the individual torsions and disulfide .pkl,
    builds the classlist structures.

    *Developer's Notes:*
    The .pkl files needed to instantiate this class and save it into its final .pkl file are
    defined in the proteusPy.data class and should not be changed. Upon initialization the class
    will load them and initialize itself.

    Example:
    >>> import proteusPy
    >>> from proteusPy.Disulfide import Disulfide
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList

    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
    >>> SS1 = PDB_SS[0]
    >>> SS1
    <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>

    Accessing by PDB_ID returns a list of Disulfides:
    >>> SS2 = PDB_SS['4yys']
    >>> SS2
    [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]

    Accessing individual disulfides by their name:
    >>> SS3 = PDB_SS['4yys_56A_98A']
    >>> SS3
    <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>

    Finally, we can access disulfides by regular slicing:
    >>> SSlist = PDB_SS[:4]
    """

    def __init__(
        self,
        verbose: bool = True,
        datadir: str = REPO_DATA_DIR,
        picklefile: str = SS_PICKLE_FILE,
        pickle_dict_file: str = SS_DICT_PICKLE_FILE,
        torsion_file: str = SS_TORSIONS_FILE,
        quiet: bool = True,
        subset: bool = False,
        cutoff: float = -1.0,
    ) -> None:
        """
        Initializing the class initiates loading either the entire Disulfide dataset,
        or the 'subset', which consists of the first 1000 PDB structures. The subset
        is useful for testing and debugging since it doesn't require nearly as much
        memory or time. The name for the subset file is hard-coded. One can pass a
        different data directory and file names for the pickle files. These different
        directories are normally established with the proteusPy.Disulfide.Extract_Disulfides
        function.
        """

        self.ModelDir = datadir
        self.PickleFile = f"{datadir}{picklefile}"
        self.PickleDictFile = f"{datadir}{pickle_dict_file}"
        self.PickleClassFile = f"{datadir}{SS_CLASS_DICT_FILE}"
        self.TorsionFile = f"{datadir}{torsion_file}"
        self.SSList = DisulfideList([], "ALL_PDB_SS")
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []
        self.QUIET = quiet

        self.tclass = None  # disulfideClass_constructor to manage classes
        self.cutoff = cutoff  # distance cutoff used to bulid the database
        self.verbose = verbose
        self.timestamp = time.time()
        self.version = proteusPy.__version__

        idlist = []

        if subset:
            self.PickleFile = f"{datadir}{SS_SUBSET_PICKLE_FILE}"
            self.PickleDictFile = f"{datadir}{SS_SUBSET_DICT_PICKLE_FILE}"
            self.TorsionFile = f"{datadir}{SS_SUBSET_TORSIONS_FILE}"

        if self.verbose:
            print(
                f"-> DisulfideLoader(): Reading disulfides from: {self.PickleFile}... ",
                end="",
            )

        with open(self.PickleFile, "rb") as f:
            # sslist = pd.compat.pickle_compat.load(f)
            sslist = pickle.load(f)
            self.SSList = sslist
            self.TotalDisulfides = len(self.SSList)

        if self.verbose:
            print(
                f"done.",
            )

        if self.verbose:
            print(
                f"-> DisulfideLoader(): Reading disulfide dict from: {self.PickleDictFile}...",
                end="",
            )

        with open(self.PickleDictFile, "rb") as f:

            self.SSDict = pickle.load(f)
            # self.SSDict = pd.compat.pickle_compat.load(f)

            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)

        if self.verbose:
            print(f"done.")

        if self.verbose:
            print(
                f"-> DisulfideLoader(): Reading Torsion DF from: {self.TorsionFile}...",
                end="",
            )

        tmpDF = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()
        self.TotalDisulfides = len(self.SSList)

        if self.verbose:
            print(f" done.")

        self.tclass = DisulfideClass_Constructor(self, self.verbose)

        if self.verbose:
            print(f"-> DisulfideLoader(): Loading complete.")
            self.describe()
        return

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

        if isinstance(item, slice):
            indices = range(*item.indices(len(self.SSList)))
            name = self.SSList[0].pdb_id
            resolution = self.SSList[0].resolution
            sublist = [self.SSList[i] for i in indices]
            return DisulfideList(sublist, name, resolution)

        if isinstance(item, int):
            if item < 0 or item >= self.TotalDisulfides:
                mess = f"DisulfideLoader(): Index {item} out of range 0-{self.TotalDisulfides - 1}"
                raise DisulfideException(mess)
            else:
                return self.SSList[item]

        try:
            # PDB_SS['4yys'] return a list of SS
            indices = self.SSDict[item]
            res = DisulfideList([], item)
            sslist = self.SSList
            for ind in indices:
                res.append(sslist[ind])
            res.resolution = res[0].resolution

        except KeyError:
            try:
                res = self.SSList.get_by_name(item)  # full disulfide name
            except:
                mess = f"DisulfideLoader(): Cannot find key {item} in SSBond dict!"
                raise DisulfideException(mess)
        return res

    def __setitem__(self, index, item):
        self.SSList[index] = self._validate_ss(item)

    def _validate_ss(self, value):
        if isinstance(value, (Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")

    @property
    def Average_Resolution(self) -> float:
        """
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        """
        res = 0.0
        cnt = 1
        sslist = self.SSList

        for ss in sslist:
            _res = ss.resolution
            if _res is not None and res != -1.0:
                res += _res
                cnt += 1
        return res / cnt

    def build_ss_from_idlist(self, idlist):
        """
        Given a list of PDBid, return a DisulfideList of Disulfides

        :param idlist: List of PDBIDs, e.g. ['4yys', '2q7q']
        :return: ProteusPy.DisulfideList.DisulfideList of ProteusPy.Disulfide.Disulfide
        """
        res = DisulfideList([], "tmp")

        for id in idlist:
            for ss in self.SSList:
                if ss.pdb_id == id:
                    res.append(ss)
                    break
        return res

    def copy(self):
        """
        Return a copy of self.

        :return: Copy of self
        """
        return copy.deepcopy(self)

    def extract_class(self, clsid) -> DisulfideList:
        """
        Return the list of disulfides corresponding to the input `clsid`.

        :param clsid: The class name to extract.
        :return: The list of disulfide bonds from the class.
        """

        # from tqdm import tqdm
        six = self.tclass.sixclass_df
        tot_classes = six.shape[0]
        class_disulfides = DisulfideList([], clsid, quiet=True)
        _pbar = tqdm(six.iterrows(), total=tot_classes, leave=True)
        for idx, row in _pbar:
            _cls = row["class_id"]
            if _cls == clsid:
                ss_list = row["ss_id"]
                pbar = tqdm(ss_list, leave=True)
                for ssid in pbar:
                    class_disulfides.append(self[ssid])
                pbar.set_postfix({"Done": ""})
                break

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
        Returns the Disulfide with the given name from the list.
        """
        for ss in self.SSList.data:
            if ss.name == name:
                return ss  # or ss.copy() !!!
        return None

    def describe(self) -> None:
        """
         Provides information about the Disulfide database contained in ```self```.

         Example:

         from proteusPy.DisulfideLoader import Load_PDB_SS
         PDB_SS = Load_PDB_SS(verbose=False, subset=False)
         PDB_SS.describe()
             =========== RCSB Disulfide Database Summary ==============
                =========== Built: 2024-02-12 17:48:13 ==============
        PDB IDs present:                    35818
        Disulfides loaded:                  120494
        Average structure resolution:       2.34 Å
        Lowest Energy Disulfide:            2q7q_75D_140D
        Highest Energy Disulfide:           1toz_456A_467A
        Cα distance cutoff:                 8.00 Å
        Total RAM Used:                     30.72 GB.
            ================= proteusPy: 0.91 =======================

        """
        vers = self.version
        tot = self.TotalDisulfides
        pdbs = len(self.SSDict)
        ram = (
            sys.getsizeof(self.SSList)
            + sys.getsizeof(self.SSDict)
            + sys.getsizeof(self.TorsionDF)
        ) / (1024 * 1024)
        res = self.Average_Resolution
        cutoff = self.cutoff
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        ssMin, ssMax = self.SSList.minmax_energy

        print(f"    =========== RCSB Disulfide Database Summary ==============")
        print(f"       =========== Built: {timestr} ==============")
        print(f"PDB IDs present:                    {pdbs}")
        print(f"Disulfides loaded:                  {tot}")
        print(f"Average structure resolution:       {res:.2f} Å")
        print(f"Lowest Energy Disulfide:            {ssMin.name}")
        print(f"Highest Energy Disulfide:           {ssMax.name}")
        print(f"Cα distance cutoff:                 {cutoff:.2f} Å")
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
        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import DisulfideLoader
        >>> from proteusPy.DisulfideList import DisulfideList

        Instantiate the Loader with the SS database subset.

        >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)

        Display the Disulfides from the PDB ID ```4yys```, overlaid onto
        a common reference (the proximal disulfides).

        >>> PDB_SS.display_overlay('4yys')

        You can also slice the loader and display as an overly.
        >>> PDB_SS[:8].display_overlay()

        """

        ssbonds = self[pdbid]
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
        >>> from proteusPy.DisulfideLoader import DisulfideLoader
        >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
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
        for k, v in enumerate(self.classdict):
            print(f"Class: |{k}|, |{v}|")

    @property
    def quiet(self) -> bool:
        """
        The loader quiet state

        :return: quiet parameter
        :rtype: bool
        """
        return self.QUIET

    @quiet.setter
    def quiet(self, perm: bool) -> None:
        """
        Sets the quiet attribute for the loader. This silences many of the BIO.PDB warnings.

        :param perm: True or False
        :type perm: bool
        """
        self.QUIET = perm

    def plot_classes_vs_cutoff(self, cutoff, steps) -> None:
        """
        Plot the total percentage and number of members for each class against the cutoff value.

        :param cutoff: Percent cutoff value for filtering the classes.
        :return: None
        """

        import matplotlib.pyplot as plt
        import numpy as np

        _cutoff = np.linspace(0, cutoff, steps)
        tot_list = []
        members_list = []

        for c in _cutoff:
            class_df = self.tclass.filter_sixclass_by_percentage(c)
            tot = class_df["percentage"].sum()
            tot_list.append(tot)
            members_list.append(class_df.shape[0])
            print(
                f"Cutoff: {c:5.3} accounts for {tot:7.2f}% and is {class_df.shape[0]:5} members long."
            )

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(_cutoff, tot_list, label="Total percentage", color="blue")
        ax2.plot(_cutoff, members_list, label="Number of members", color="red")

        ax1.set_xlabel("Cutoff")
        ax1.set_ylabel("Total percentage", color="blue")
        ax2.set_ylabel("Number of members", color="red")

        plt.show()

    def plot_binary_to_sixclass_incidence(
        self, light=True, save=False, savedir="."
    ) -> None:
        """
        Plot the incidence of all sextant Disulfide classes for a given binary class.

        :param loader: `proteusPy.DisulfideLoader` object
        """

        from proteusPy.DisulfideClasses import plot_count_vs_class_df

        def _enumerate_sixclass_fromlist(sslist):
            x = []
            y = []

            for sixcls in sslist:
                if sixcls is not None:
                    _y = self.tclass.sslist_from_classid(sixcls)
                    # it's possible to have 0 SS in a class
                    if _y is not None:
                        # only append if we have both.
                        x.append(sixcls)
                        y.append(len(_y))

            sslist_df = pd.DataFrame(columns=["class_id", "count"])
            sslist_df["class_id"] = x
            sslist_df["count"] = y
            return sslist_df

        clslist = self.tclass.classdf["class_id"]
        for cls in clslist:
            sixcls = self.tclass.binary_to_six_class(cls)
            df = _enumerate_sixclass_fromlist(sixcls)
            plot_count_vs_class_df(df, cls, theme="light", save=save, savedir=savedir)
        return

    def enumerate_sixclass_fromlist(self, sslist) -> pd.DataFrame:
        x = []
        y = []

        for sixcls in sslist:
            if sixcls is not None:
                _y = self.tclass.sslist_from_classid(sixcls)
                # it's possible to have 0 SS in a class
                if _y is not None:
                    # only append if we have both.
                    x.append(sixcls)
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
        self.version = proteusPy.__version__
        self.cutoff = cutoff

        if subset:
            fname = LOADER_SUBSET_FNAME
        else:
            fname = LOADER_FNAME

        _fname = f"{savepath}{fname}"

        if self.verbose:
            print(f"-> DisulfideLoader.save(): Writing {_fname}... ")

        with open(_fname, "wb+") as f:
            pickle.dump(self, f)

        if self.verbose:
            print(f"-> DisulfideLoader.save(): Done.")


# class ends


def Download_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False):
    """
    Download the databases from my Google Drive.

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to False
    """

    import gdown

    _fname_sub = f"{loadpath}{LOADER_SUBSET_FNAME}"
    _fname_all = f"{loadpath}{LOADER_FNAME}"

    if verbose:
        print(f"--> DisulfideLoader: Downloading Disulfide Database from Drive...")

    gdown.download(LOADER_ALL_URL, _fname_all, quiet=False)

    if subset:
        if verbose:
            print(
                f"--> DisulfideLoader: Downloading Disulfide Subset Database from Drive..."
            )

        gdown.download(LOADER_SUBSET_URL, _fname_sub, quiet=False)

    return


def Download_PDB_SS_GitHub(loadpath=DATA_DIR, verbose=True, subset=False):
    """
    Download the databases from Github. Note: if you change the database these sizes will
    need to be changed!

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to True
    """

    import urllib

    _good1 = 0  # all data
    _good2 = 0  # subset data

    _fname_sub = f"{loadpath}{LOADER_SUBSET_FNAME}"
    _fname_all = f"{loadpath}{LOADER_FNAME}"

    _all_length = 340371775
    _subset_length = 9636086

    if verbose:
        print(f"--> DisulfideLoader: Downloading Disulfide Database from GitHub...")

    resp, headers = urllib.request.urlretrieve(
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
                f"--> DisulfideLoader: Downloading Disulfide Subset Database from GitHub..."
            )

        resp, headers = urllib.request.urlretrieve(
            "https://github.com/suchanek/proteusPy/raw/master/data/PDB_SS_SUBSET_LOADER.pkl",
            _fname_sub,
        )
        num_bytes = headers.get("content-length")
        if num_bytes == _subset_length:
            _good2 = 1
        else:
            print(f"--> Read: {num_bytes}, expecting: {_subset_length}")
    return _good1 + _good2


def Load_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False) -> DisulfideLoader:
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

    _fname_sub = f"{loadpath}{LOADER_SUBSET_FNAME}"
    _fname_all = f"{loadpath}{LOADER_FNAME}"

    if subset:
        _fname = _fname_sub
    else:
        _fname = _fname_all

    if not os.path.exists(_fname_sub):
        res2 = Download_PDB_SS(loadpath=loadpath, verbose=verbose, subset=True)

    if not os.path.exists(_fname_all):
        res2 = Download_PDB_SS(loadpath=loadpath, verbose=verbose, subset=False)

    # first attempt to read the local copy of the loader
    if verbose:
        print(f"-> load_PDB_SS(): Reading {_fname}... ")

    with open(_fname, "rb") as f:
        res = pickle.load(f)
        # res = pd.compat.pickle_compat.load(f)

    if verbose:
        print(f"-> load_PDB_SS(): Done reading {_fname}... ")
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
