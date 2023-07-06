'''
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Author: Eric G. Suchanek, PhD
Last revision: 7/2/2023
'''

# Author: Eric G. Suchanek, PhD.
# Last modification: 2/20/23
# Cα N, Cα, Cβ, C', Sγ Å °

import sys
import copy
from io import StringIO
import time
import matplotlib.pyplot as plt

import pandas as pd
import pyvista as pv
import pickle
import numpy as np

import proteusPy
from proteusPy.ProteusGlobals import PDB_DIR, MODEL_DIR,  REPO_DATA_DIR
from proteusPy.atoms import *

from proteusPy.data import SS_PICKLE_FILE, SS_TORSIONS_FILE, SS_DICT_PICKLE_FILE, DATA_DIR
from proteusPy.data import LOADER_FNAME, LOADER_SUBSET_FNAME, LOADER_ALL_URL, LOADER_SUBSET_URL

from proteusPy.DisulfideList import DisulfideList
from proteusPy.Disulfide import Disulfide

from proteusPy.DisulfideExceptions import *
from proteusPy.data import *
from proteusPy.DisulfideClass_Constructor import DisulfideClass_Constructor

try:
    # Check if running in Jupyter
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

# Now use tqdm as normal, depending on your environment

class DisulfideLoader:
    '''
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
    >>> SS1 = DisulfideList([],'tmp1')
    >>> SS2 = DisulfideList([],'tmp2')
    
    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)

    Accessing by index value:
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
    '''

    def __init__(self, verbose=True, datadir=REPO_DATA_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                torsion_file=SS_TORSIONS_FILE, quiet=True, 
                subset=False,
                cutoff=-1.0):
        '''
        Initializing the class initiates loading either the entire Disulfide dataset,
        or the 'subset', which consists of the first 1000 PDB structures. The subset
        is useful for testing and debugging since it doesn't require nearly as much
        memory or time. The name for the subset file is hard-coded. One can pass a
        different data directory and file names for the pickle files. These different
        directories are normally established with the proteusPy.Disulfide.Extract_Disulfides 
        function.
        '''

        self.ModelDir = datadir
        self.PickleFile = f'{datadir}{picklefile}'
        self.PickleDictFile = f'{datadir}{pickle_dict_file}'
        self.PickleClassFile = f'{datadir}{SS_CLASS_DICT_FILE}'
        self.TorsionFile = f'{datadir}{torsion_file}'
        self.SSList = DisulfideList([], 'ALL_PDB_SS')
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []
        self.QUIET = quiet
        
        self.tclass = None        # disulfideClass_constructor to manage classes
        self.cutoff = cutoff      # distance cutoff used to bulid the database
        self.verbose = verbose
        self.timestamp = time.time()
        self.version = proteusPy.__version__

        idlist = []

        if subset:
            self.PickleFile = f'{datadir}{SS_SUBSET_PICKLE_FILE}'
            self.PickleDictFile = f'{datadir}{SS_SUBSET_DICT_PICKLE_FILE}'
            self.TorsionFile = f'{datadir}{SS_SUBSET_TORSIONS_FILE}'
        
        if self.verbose:
            print(f'-> DisulfideLoader(): Reading disulfides from: {self.PickleFile}... ', end='')
    
        with open(self.PickleFile, 'rb') as f:
            sslist = pickle.load(f)
            self.SSList = sslist
            self.TotalDisulfides = len(self.SSList)

        if self.verbose:
            print(f'done.',)

        if self.verbose:
            print(f'-> DisulfideLoader(): Reading disulfide dict from: {self.PickleDictFile}...', end='')
    
        with open(self.PickleDictFile, 'rb') as f:
            self.SSDict = pickle.load(f)
            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)
    
        if self.verbose:
            print(f'done.')

        if self.verbose:
            print(f'-> DisulfideLoader(): Reading Torsion DF from: {self.TorsionFile}...', end='')

        tmpDF  = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()
        self.TotalDisulfides = len(self.SSList)

        if self.verbose:
            print(f' done.')

        self.tclass = DisulfideClass_Constructor(self, self.verbose)

        if self.verbose:    
            print(f'-> DisulfideLoader(): Loading complete.')
            self.describe()
        return

    # 
    # overload __getitem__ to handle slicing and indexing, and access by name
    def __getitem__(self, item):
        res = DisulfideList([], 'none')

        if isinstance(item, slice):
            indices = range(*item.indices(len(self.SSList)))
            name = self.SSList[0].pdb_id
            resolution = self.SSList[0].resolution
            sublist = [self.SSList[i] for i in indices]
            return DisulfideList(sublist, name, resolution)
        
        if isinstance(item, int):
            if (item < 0 or item >= self.TotalDisulfides):
                mess = f'DisulfideLoader(): Index {item} out of range 0-{self.TotalDisulfides - 1}'
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
                res = self.SSList.get_by_name(item) # full disulfide name
            except: 
                mess = f'DisulfideLoader(): Cannot find key {item} in SSBond dict!'
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
        '''
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        '''
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
        '''
        Given a list of PDBid, return a DisulfideList of Disulfides

        :param idlist: List of PDBIDs, e.g. ['4yys', '2q7q']
        :return: ProteusPy.DisulfideList.DisulfideList of ProteusPy.Disulfide.Disulfide
        '''
        res = DisulfideList([], 'tmp')
        
        for id in idlist:
            for ss in self.SSList:
                if ss.pdb_id == id:
                    res.append(ss)
                    break
        return res
    
    def copy(self):
        '''
        Return a copy of self.

        :return: Copy of self
        '''
        return copy.deepcopy(self)

    def extract_class(self, clsid):
        """
        Return the list of disulfides corresponding to the input `clsid`.
    
        :param clsid: The class name to extract.
        :return: The list of disulfide bonds from the class.
        """
        
        from tqdm import tqdm
        six = self.tclass.sixclass_df
        tot_classes = six.shape[0]
        class_disulfides = DisulfideList([], clsid, quiet=True)
        _pbar = tqdm(six.iterrows(), total=tot_classes, leave=True)
        for idx, row in _pbar:
            _cls = row['class_id']
            if _cls == clsid:
                ss_list = row['ss_id']
                pbar = tqdm(ss_list, leave=True)
                for ssid in pbar:
                    class_disulfides.append(self[ssid])
                pbar.set_postfix({'Done': ''})
                break

            _pbar.set_postfix({'Cnt': idx})
            
        return class_disulfides
    
    def getlist(self) -> DisulfideList:
        '''
        Return the list of Disulfides contained in the class.

        :return: DisulfideList
        :rtype: DisulfideList
        '''
        return copy.deepcopy(self.SSList)
    
    def get_by_name(self, name) -> Disulfide:
        '''
        Returns the Disulfide with the given name from the list.
        '''
        for ss in self.SSList.data:
            if ss.name == name:
                return ss  # or ss.copy() !!!
        return None
    def describe(self):
        '''
        Provides information about the Disulfide database contained in ```self```.

        Example:
        >>> from proteusPy.DisulfideLoader import Load_PDB_SS
        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=False)
        >>> PDB_SS.describe()
            =========== RCSB Disulfide Database Summary ==============
        PDB IDs present:                    35818
        Disulfides loaded:                  120697
        Average structure resolution:       2.34 Å
        Lowest Energy Disulfide:            2q7q_75D_140D
        Highest Energy Disulfide:           1toz_456A_467A
        Total RAM Used:                     29.26 GB.

        '''
        vers = self.version
        tot = self.TotalDisulfides
        pdbs = len(self.SSDict)
        ram = (sys.getsizeof(self.SSList) + sys.getsizeof(self.SSDict) + sys.getsizeof(self.TorsionDF)) / (1024 * 1024)
        res = self.Average_Resolution
        cutoff = self.cutoff
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        ssMin, ssMax = self.SSList.minmax_energy()
        
        print(f'    =========== RCSB Disulfide Database Summary ==============')
        print(f'       =========== Built: {timestr} ==============')
        print(f'PDB IDs present:                    {pdbs}')
        print(f'Disulfides loaded:                  {tot}')
        print(f'Average structure resolution:       {res:.2f} Å')
        print(f'Lowest Energy Disulfide:            {ssMin.name}')
        print(f'Highest Energy Disulfide:           {ssMax.name}')
        print(f'Ca distance cutoff:                 {cutoff:.2f} Å')
        print(f'Total RAM Used:                     {ram:.2f} GB.')
        print(f'    ================= proteusPy: {vers} =======================')
       
    def display_overlay(self, pdbid):
        ''' 
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

        ''' 

        ssbonds = self[pdbid]
        ssbonds.display_overlay()
        return
    
    
    def getTorsions(self, pdbID=None) -> pd.DataFrame:
        '''
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
        >>> Tor_DF.describe()
                  proximal       distal         chi1         chi2         chi3         chi4  ...  ca_distance     phi_prox     psi_prox     phi_dist     psi_dist  torsion_length
        count  3393.000000  3393.000000  3393.000000  3393.000000  3393.000000  3393.000000  ...  3393.000000  3393.000000  3393.000000  3393.000000  3393.000000     3393.000000
        mean    231.513999   280.056587   -49.134436   -15.616297    -3.727982   -31.909496  ...     5.554563   -98.491326    63.029854   -95.890117    62.408772      225.242256
        std     292.300344   293.503989    95.456284   104.318483    93.894477   103.553641  ...     1.489138    44.275097    99.266921    44.796764    97.231632       53.309336
        min       1.000000     6.000000  -179.947368  -179.990782  -179.081812  -179.940602  ...     2.941898  -180.000000  -180.000000  -180.000000  -180.000000      116.478788
        25%      48.000000    96.000000   -83.281768   -88.455019   -87.763001   -95.240905  ...     5.075491  -129.605300   -26.792543  -123.131231   -25.503215      181.929048
        50%     136.000000   194.000000   -63.878076   -59.597437   -64.977491   -69.607516  ...     5.612046   -97.569676   112.772998   -97.956322   112.956483      225.617493
        75%     310.000000   361.000000   -47.459809    82.078365    94.169603    69.488862  ...     6.112471   -69.670087   143.781003   -70.485076   142.808862      263.003492
        max    2592.000000  2599.000000   179.918814   179.987671   179.554652   179.977181  ...    75.611323   177.021502   179.856474   178.886602   179.735964      368.022621
        <BLANKLINE>
        [8 rows x 14 columns]
        '''
        res_df = pd.DataFrame()

        if pdbID:
            try:
                res = self.SSDict[pdbID]
                sel = self.TorsionDF['source'] == pdbID
                res_df = self.TorsionDF[sel]
                return res_df.copy()
            except KeyError:
                mess = f'! Cannot find key {pdbID} in SSBond DB'
                raise DisulfideParseWarning(mess)
        else:
            return copy.deepcopy(self.TorsionDF)
    
    def list_binary_classes(self):
        for k,v in enumerate(self.classdict):
            print(f'Class: |{k}|, |{v}|')

    @property
    def quiet(self) -> bool:
        '''
        The loader quiet state

        :return: quiet parameter
        :rtype: bool
        '''
        return self.QUIET

    @quiet.setter
    def quiet(self, perm: bool) -> None:
        '''
        Sets the quiet attribute for the loader. This silences many of the BIO.PDB warnings.

        :param perm: True or False
        :type perm: bool
        '''
        self.QUIET = perm
    
    def plot_classes_vs_cutoff(self, cutoff, steps):
        """
        Plot the total percentage and number of members for each class against the cutoff value.
        
        :param cutoff: Percent cutoff value for filtering the classes.
        :return: None
        """
        _cutoff = np.linspace(0, cutoff, steps)
        tot_list = []
        members_list = []

        for c in _cutoff:
            class_df = self.tclass.filter_sixclass_by_percentage(c)
            tot = class_df['percentage'].sum()
            tot_list.append(tot)
            members_list.append(class_df.shape[0])
            print(f'Cutoff: {c:5.3} accounts for {tot:7.2f}% and is {class_df.shape[0]:5} members long.')

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(_cutoff, tot_list, label='Total percentage', color='blue')
        ax2.plot(_cutoff, members_list, label='Number of members', color='red')

        ax1.set_xlabel('Cutoff')
        ax1.set_ylabel('Total percentage', color='blue')
        ax2.set_ylabel('Number of members', color='red')

        plt.show()

    def plot_binary_to_sixclass_incidence(self, light=True, save=False, savedir='.'):
        '''
        Plot the incidence of all sextant Disulfide classes for a given binary class.

        :param loader: `proteusPy.DisulfideLoader` object
        '''
        
        from proteusPy.DisulfideClasses import plot_count_vs_class_df

        def enumerate_sixclass_fromlist(sslist):
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

            sslist_df = pd.DataFrame(columns=['class_id', 'count'])
            sslist_df['class_id'] = x
            sslist_df['count'] = y
            return(sslist_df)

        clslist = self.tclass.classdf['class_id']
        for cls in clslist:
            sixcls = self.tclass.binary_to_six_class(cls)
            df = enumerate_sixclass_fromlist(sixcls)
            plot_count_vs_class_df(df, cls, theme='light', save=save, savedir=savedir)
        return

    def save(self, savepath=DATA_DIR, subset=False, cutoff=-1.0):
        '''
        Save a copy of the fully instantiated Loader to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        :param fname: Filename, defaults to LOADER_FNAME
        :param verbose: Verbosity, defaults to False
        :param cutoff: Distance cutoff used to build the database, -1 means no cutoff.
        '''
        self.version = proteusPy.__version__
        self.cutoff = cutoff

        if subset:
            fname = LOADER_SUBSET_FNAME
        else:
            fname = LOADER_FNAME

        _fname = f'{savepath}{fname}'

        if self.verbose:
            print(f'-> DisulfideLoader.save(): Writing {_fname}... ')
        
        with open(_fname, 'wb+') as f:
            pickle.dump(self, f)
        
        if self.verbose:
            print(f'-> DisulfideLoader.save(): Done.')
   
# class ends

def Download_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False):
    '''
    Download the databases from Google Drive.

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to False
    '''
    # normally the .pkl files are local, EXCEPT for the first run from a newly-installed proteusPy 
    # distribution. In that case we need to download the files for all disulfides and the subset
    # from the Google Drive storage.
    
    import gdown
    
    _good1 = 0 # all data
    _good2 = 0 # subset data
    
    _fname_sub = f'{loadpath}{LOADER_SUBSET_FNAME}'
    _fname_all = f'{loadpath}{LOADER_FNAME}'
    
    
    if verbose:
        print(f'-> Download_PDB_SS(): Reading disulfides from Google Drive... ')
    
    if subset:
        if gdown.download(LOADER_SUBSET_URL, _fname_sub, quiet=False) is not None:
            os.sync()
            if os.path.exists(_fname_sub):
                _good2 = 2
    else:
        if gdown.download(LOADER_ALL_URL, _fname_all, quiet=False) is not None:
            os.sync()
            if os.path.exists(_fname_all):
                _good1 = 1

    return _good1 + _good2


def Load_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False) -> DisulfideLoader:
    '''
    Load the fully instantiated Disulfide database from the specified file. Use the
    defaults unless you are building the database by hand. *This is the function
    used to load the built database.*

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param verbose: Verbosity, defaults to False
    :param subset: If True, load the subset DB, otherwise load the full database
    '''
    # normally the .pkl files are local, EXCEPT for the first run from a newly-installed proteusPy 
    # distribution. In that case we need to download the files for all disulfides and the subset
    # from the Google Drive storage.
    
    import gdown
    
    _good1 = False # all data
    _good2 = False # subset data
    
    _fname_sub = f'{loadpath}{LOADER_SUBSET_FNAME}'
    _fname_all = f'{loadpath}{LOADER_FNAME}'
    
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
        print(f'-> load_PDB_SS(): Reading {_fname}... ')


    with open(_fname, 'rb') as f:
        res = pickle.load(f)
    if verbose:
        print(f'-> load_PDB_SS(): Done reading {_fname}... ')
    return res


# End of file
