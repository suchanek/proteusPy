'''
This module is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds.
This work is based on the original C/C++ implementation by Eric G. Suchanek. \n

Author: Eric G. Suchanek, PhD
Last revision: 3/5/2023
'''

# Author: Eric G. Suchanek, PhD.
# Last modification: 2/20/23
# Cα N, Cα, Cβ, C', Sγ Å °

__pdoc__ = {'__all__': True, '__getitem__': True}

import sys
import copy
from io import StringIO
import time
import datetime

import pandas as pd
import pyvista as pv
import pickle
import numpy as np

import proteusPy
from proteusPy.ProteusGlobals import PDB_DIR, MODEL_DIR
from proteusPy.atoms import *

from proteusPy.data import SS_PICKLE_FILE, SS_TORSIONS_FILE, SS_DICT_PICKLE_FILE, DATA_DIR
from proteusPy.data import LOADER_FNAME, LOADER_SUBSET_FNAME

from proteusPy.DisulfideList import DisulfideList
from proteusPy.Disulfide import Torsion_DF_Cols
from proteusPy.Disulfide import Disulfide

from proteusPy.DisulfideExceptions import *
from proteusPy.data import *
from proteusPy.DisulfideClasses import create_six_class_df, create_classes

import itertools

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
    >>> SSlist.display(style='sb') 
    '''

    def __init__(self, verbose=True, datadir=DATA_DIR, picklefile=SS_PICKLE_FILE, 
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
        self.classdict = {}
        self.classdf = None
        self.sixclass_df = None
        self.cutoff = cutoff
        self.verbose = verbose
        self.timestamp = time.time()
        self.version = 'V0.0'

        idlist = []

        if subset:
            self.PickleFile = f'{datadir}{SS_SUBSET_PICKLE_FILE}'
            self.PickleDictFile = f'{datadir}{SS_SUBSET_DICT_PICKLE_FILE}'
            self.TorsionFile = f'{datadir}{SS_SUBSET_TORSIONS_FILE}'
        
        if self.verbose:
            print(f'--> DisulfideLoader(): Reading disulfides from: {self.PickleFile}... ', end='')
        
        with open(self.PickleFile, 'rb') as f:
            sslist = pickle.load(f)
            self.SSList = sslist

        if self.verbose:
            print(f'done.',)

        self.TotalDisulfides = len(self.SSList)
        
        if self.verbose:
            print(f'--> DisulfideLoader(): Reading disulfide dict2 from: {self.PickleDictFile}...', end='')
        
        with open(self.PickleDictFile, 'rb') as f:
            self.SSDict = pickle.load(f)
            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)

        if self.verbose:
            print(f'done.')

        if self.verbose:
            print(f'--> DisulfideLoader(): Reading Torsion DF from: {self.TorsionFile}...', end='')

        tmpDF  = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()

        if self.verbose:
            print(f' done.')

        self.build_classes()

        if self.verbose:    
            print(f'--> DisulfideLoader(): Loading complete.')
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
    
    def build_classes(self):
        '''
        Build the internal dictionary mapping disulfides to class names.

        Disulfide classes are defined using the ± formalism described by 
        Hogg et al. (Biochem, 2006, 45, 7429-7433), across all 32 (2^5), possible 
        binary sidechain torsional combinations. Classes are named per Hogg's convention.
        The ``class_id`` represents the sign of each dihedral angle $\chi_{1} - \chi_{1'}$
        where *0* repreents *negative* dihedral angle and *2* a *positive* angle.

        |   class_id | SS_Classname   | FXN        |   count |   incidence |
        |-----------:|:---------------|:-----------|--------:|------------:|
        |      00000 | -LHSpiral      | UNK        |   31513 |  0.261092   |
        |      00002 | 00002          | UNK        |    5805 |  0.0480956  |
        |      00020 | -LHHook        | UNK        |    3413 |  0.0282774  |
        |      00022 | 00022          | UNK        |    1940 |  0.0160733  |
        |      00200 | -RHStaple      | Allosteric |   12735 |  0.105512   |
        |      00202 | 00202          | UNK        |     993 |  0.00822721 |
        |      00220 | 00220          | UNK        |    5674 |  0.0470103  |
        |      00222 | 00222          | UNK        |    5092 |  0.0421883  |
        |      02000 | 02000          | UNK        |    4749 |  0.0393465  |
        |      02002 | 02002          | UNK        |    3774 |  0.0312684  |
        |      02020 | -LHStaple      | UNK        |    1494 |  0.0123781  |
        |      02022 | 02022          | UNK        |     591 |  0.00489656 |
        |      02200 | -RHHook        | UNK        |    5090 |  0.0421717  |
        |      02202 | 02202          | UNK        |     533 |  0.00441602 |
        |      02220 | -RHSpiral      | UNK        |    6751 |  0.0559335  |
        |      02222 | 02222          | UNK        |    3474 |  0.0287828  |
        |      20000 | ±LHSpiral      | UNK        |    3847 |  0.0318732  |
        |      20002 | +LHSpiral      | UNK        |     875 |  0.00724956 |
        |      20020 | ±LHHook        | UNK        |     803 |  0.00665302 |
        |      20022 | +LHHook        | UNK        |     602 |  0.0049877  |
        |      20200 | ±RHStaple      | UNK        |     419 |  0.0034715  |
        |      20202 | +RHStaple      | UNK        |     293 |  0.00242757 |
        |      20220 | ±RHHook        | Catalytic  |    1435 |  0.0118893  |
        |      20222 | 20222          | UNK        |     488 |  0.00404318 |
        |      22000 | -/+LHHook      | UNK        |    2455 |  0.0203402  |
        |      22002 | 22002          | UNK        |    1027 |  0.00850891 |
        |      22020 | ±LHStaple      | UNK        |    1046 |  0.00866633 |
        |      22022 | +LHStaple      | UNK        |     300 |  0.00248556 |
        |      22200 | -/+RHHook      | UNK        |    6684 |  0.0553783  |
        |      22202 | +RHHook        | UNK        |     593 |  0.00491313 |
        |      22220 | ±RHSpiral      | UNK        |    2544 |  0.0210776  |
        |      22222 | +RHSpiral      | UNK        |    3665 |  0.0303653  |
        '''

        from proteusPy.DisulfideClasses import create_classes

        def ss_id_dict(df):
            ss_id_dict = dict(zip(df['SS_Classname'], df['ss_id']))
            return ss_id_dict

        tors_df = self.getTorsions()

        if self.verbose:
            print(f'-> build_classes(): creating SS classes...')

        grouped = create_classes(tors_df)

        # grouped.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')
        #if self.verbose:
        #    print(f'{grouped.head(32)}')
        
        class_df = pd.read_csv(StringIO(SS_CLASS_DEFINITIONS), dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
        class_df['FXN'].str.strip()
        class_df['SS_Classname'].str.strip()
        class_df['class_id'].str.strip()

        if self.verbose:
            print(f'--> build_classes(): merging...')

        merged = self.concat_dataframes(class_df, grouped)
        merged.drop(columns=['Idx'], inplace=True)

        self.classdf = merged.copy()
        self.classdict = ss_id_dict(merged)

        if self.verbose:
            print(f'-> build_classes(): initialization complete.')
        
        return
    
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
    
    def create_six_class_df(self):
        """
        Create a new DataFrame from the input with a 6-class encoding for input 'chi' values.
        
        The function takes a pandas DataFrame containing the following columns:
        'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'ca_distance', 'cb_distance',
        'torsion_length', 'energy', and 'rho', and adds a class ID column based on the following rules:
        
        1. A new column named `class_id` is added, which is the concatenation of the individual class IDs per Chi.
        2. The DataFrame is grouped by the `class_id` column, and a new DataFrame is returned that shows the unique `ss_id` values for each group,
        the count of unique `ss_id` values, the incidence of each group as a proportion of the total DataFrame, and the
        percentage of incidence.

        :param df: A pandas DataFrame containing columns 'ss_id', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5',
                'ca_distance', 'cb_distance', 'torsion_length', 'energy', and 'rho'
        :return: The grouped DataFrame with the added class column.
        """
        from proteusPy.DisulfideClasses import get_sixth_quadrant

        _df = pd.DataFrame()
        df = self.getTorsions
        # create the chi_t columns for each chi column
        for col_name in ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']:
            _df[col_name + '_t'] = df[col_name].apply(get_sixth_quadrant)
        
        # create the class_id column
        df['class_id'] = _df[['chi1_t', 'chi2_t', 'chi3_t', 'chi4_t', 'chi5_t']].apply(lambda x: ''.join(x), axis=1)

        # group the DataFrame by class_id and return the grouped data
        grouped = df.groupby('class_id').agg({'ss_id': 'unique'})
        grouped['count'] = grouped['ss_id'].apply(lambda x: len(x))
        grouped['incidence'] = grouped['count'] / len(df)
        grouped['percentage'] = grouped['incidence'] * 100
        grouped.reset_index(inplace=True)

        self.sixclass_df = grouped

    
    def concat_dataframes(self, df1, df2):
        """
        Concatenates columns from one data frame into the other and returns the new result.

        :param df1 : pandas.DataFrame - The first data frame.
        :param df2 : pandas.DataFrame - The second data frame.

        :return: pandas.DataFrame - The concatenated data frame.
        """
        # Merge the data frames based on the 'SS_Classname' column
        result = pd.merge(df1, df2, on='class_id')

        return result

    def copy(self):
        '''
        Return a copy of self

        :return: Copy of self
        :rtype: proteusPy.DisulfideLoader
        '''
        return copy.deepcopy(self)

    def getlist(self) -> DisulfideList:
        '''
        Return the list of Disulfides contained in the class.

        :return: DisulfideList
        :rtype: DisulfideList
        '''
        return copy.deepcopy(self.SSList)
    
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
    
    def from_class(self, classid: str) -> DisulfideList:
        '''
        Return a list of disulfides corresponding to the input class ID
        string.

        :param classid: Class ID, e.g. '+RHStaple'
        :return: DisulfideList of class members
        '''
        from tqdm import tqdm
        from proteusPy.ProteusGlobals import PBAR_COLS
        res = DisulfideList([], classid)

        try:
            sslist = self.classdict[classid]
            if self.verbose:
                pbar = tqdm(sslist, ncols=PBAR_COLS)
                for ssid in pbar:
                    res.append(self[ssid])
                return res

            res = DisulfideList([self[ssid] for ssid in sslist], classid)
            return res
        except KeyError:
            print(f'No class: {classid}')
        return

    
# class ends
class DisulfideClass_Constructor():
    '''
    Class manages structural classes for the disulfide bonds contained
    in the proteusPy disulfide database
    '''

    def __init__(self, verbose=False, bootstrap=False) -> None:
        self.verbose = verbose
        self.classdict = {}
        self.classdf = None

        if bootstrap:
            if self.verbose:
                print(f'--> DisulfideClass_Constructor(): Building SS classes...')
            self.build_yourself()
        else:
            self.classdict = self.load_class_dict()

    def load_class_dict(self, fname=f'{DATA_DIR}{SS_CLASS_DICT_FILE}') -> dict:
        with open(fname,'rb') as f:
            #res = pickle.load(f)
            self.classdict = pickle.load(f)
    
    def build_class_df(self, class_df, group_df):
        ss_id_col = group_df['ss_id']
        result_df = pd.concat([class_df, ss_id_col], axis=1)
        return result_df

    def list_binary_classes(self):
        for k,v in enumerate(self.classdict):
            print(f'Class: |{k}|, |{v}|')

    #  class_cols = ['Idx','chi1_s','chi2_s','chi3_s','chi4_s','chi5_s','class_id','SS_Classname','FXN',
    # 'count','incidence','percentage','ca_distance_mean',
    # 'ca_distance_std','torsion_length_mean','torsion_length_std','energy_mean','energy_std']

    def concat_dataframes(self, df1, df2):
        """
        Concatenates columns from one data frame into the other 
        and returns the new result.

        Parameters
        ----------
        df1 : pandas.DataFrame
            The first data frame.
        df2 : pandas.DataFrame
            The second data frame.

        Returns
        -------
        pandas.DataFrame
            The concatenated data frame.

        """
        # Merge the data frames based on the 'SS_Classname' column
        result = pd.merge(df1, df2, on='class_id')

        return result

    def build_yourself(self):
        '''
        Builds the internal structures needed for the loader, including binary and six-fold classes.
        The classnames are defined by the sign of the dihedral angles, per XXX', the list of SS within
        the database classified, and the resulting dict created.
        '''

        from proteusPy.DisulfideClasses import create_classes, create_six_class_df

        def ss_id_dict(df):
            ss_id_dict = dict(zip(df['SS_Classname'], df['ss_id']))
            return ss_id_dict

        PDB_SS = Load_PDB_SS(verbose=self.verbose, subset=False)
        self.version = proteusPy.__version__

        if self.verbose:
            PDB_SS.describe()

        tors_df = PDB_SS.getTorsions()

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): creating binary SS classes...')
        grouped = create_classes(tors_df)        
        
        # grouped.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')
        
        # this file is hand made. Do not change it. -egs-
        #class_df = pd.read_csv(f'{DATA_DIR}PDB_ss_classes_master2.csv', dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})

        # !!! df = pd.read_csv(pd.compat.StringIO(csv_string))
        # class_df = pd.read_csv(f'{DATA_DIR}PDB_SS_class_definitions.csv', dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
        
        class_df = pd.read_csv(StringIO(SS_CLASS_DEFINITIONS), dtype={'class_id': 'string', 'FXN': 'string', 'SS_Classname': 'string'})
        class_df['FXN'].str.strip()
        class_df['SS_Classname'].str.strip()
        class_df['class_id'].str.strip()

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): merging...')

        merged = self.concat_dataframes(class_df, grouped)
        merged.drop(columns=['Idx'], inplace=True)

        classdict = ss_id_dict(merged)
        self.classdict = classdict

        merged.to_csv(f'{DATA_DIR}PDB_SS_merged.csv')
        self.classdf = merged.copy()

        fname = f'{DATA_DIR}{SS_CLASS_DICT_FILE}'

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): writing {fname}...')

        with open(fname, "wb+") as f:
            pickle.dump(classdict, f)

        if self.verbose:
            print(f'-> DisulfideClass_Constructor(): creating sixfold SS classes...')
        
        grouped_sixclass = create_six_class_df(tors_df)
        grouped_sixclass.to_csv(f'{DATA_DIR}PDB_ss_classes.csv')
        self.sixclass_df = grouped_sixclass

        if self.verbose:
            print(f'--> DisulfideClass_Constructor(): ')
        
        if self.verbose:
            print(f'--> DisulfideClass_Constructor(): initialization complete.')
        
        return

# class definition ends

def Load_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False) -> DisulfideLoader:
    '''
    Load the fully instantiated Disulfide database from the specified file. Use the
    defaults unless you are building the database by hand. *This is the function
    used to load the built database.*

    :param loadpath: Path from which to load, defaults to DATA_DIR
    :param fname: Filename, defaults to LOADER_FNAME
    :param verbose: Verbosity, defaults to False
    :param subset: If True, load the subset DB, otherwise load the full database
    '''
    if subset:
        _fname = f'{loadpath}{LOADER_SUBSET_FNAME}'
    else:
        _fname = f'{loadpath}{LOADER_FNAME}'

    if verbose:
        print(f'-> load_PDB_SS(): Reading {_fname}... ', end='')
    try:
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
        if verbose:
            print(f'done.')
        return res
    except:
        mess = f'-> load_PDB_SS(): cannot open file {_fname}'
        raise DisulfideIOException(mess)

