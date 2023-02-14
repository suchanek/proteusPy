'''
This class embodies functions used to load Disulfide Bonds extracted from the
RCSB Protein Databank, and initializes the DisulfideLoader with the Disulfide
lists and dictionary.

This file is part of the proteusPy package.
Author: Eric G. Suchanek, PhD.
'''

# Author: Eric G. Suchanek, PhD.
# Last modification: 2/7/23

import sys
import copy

import pandas as pd
import pyvista as pv
import pickle

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

class DisulfideLoader:
    '''
    This class represents the disulfide database itself and is the primary means of accession. 
    The entirety of the RCSB disulfide database is stored within the class via a
    proteusPy.DisulfideList.DisulfideList list, a ```Pandas``` .csv file, and a ```dict``` of
    indices mapping the PDB IDs into their respective list of disulfides. The datastructures allow
    simple, direct and flexible access to the disulfide structures contained herein. 
    This makes it possible to access the disulfides by array index, PDB structure ID or disulfide name.
    
    The class can also render Disulfides to a pyVista window using the 
    DisulfideLoader.display() method. See below for examples.\n
    
    *Developer's Notes:*
    The .pkl files needed to instantiate this class and save it into its final .pkl file are
    defined in the proteusPy.data class and should not be changed. Upon initialization the class
    will load them and initialize itself. 
    
    The class loads files created from the proteusPy.Disulfide.Extract_Disulfides() routine 
    and initializes itself with their contents. 
    

    Example:
    >>> import proteusPy
    >>> from proteusPy.Disulfide import Disulfide
    >>> from proteusPy.DisulfideLoader import DisulfideLoader
    >>> from proteusPy.DisulfideList import DisulfideList
    >>> SS1 = DisulfideList([],'tmp1')
    >>> SS2 = DisulfideList([],'tmp2')
    
    >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
    >>> SS1 = PDB_SS[0]
    >>> SS1
    <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>
    
    Accessing by PDB_ID returns a list of Disulfides:
    >>> SS2 = PDB_SS['4yys']
    >>> SS2
    [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>]
    
    One can can access individual disulfides by their name:
    >>> SS3 = PDB_SS['4yys_56A_98A']
    >>> SS3
    <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>
    
    Finally, we can access disulfides by regular slicing:
    >>> SSlist = PDB_SS[:4]
    >>> SSlist.display(style='sb') 
    '''

    def __init__(self, verbose=True, datadir=DATA_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                torsion_file=SS_TORSIONS_FILE, quiet=True, subset=False):
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
        self.TorsionFile = f'{datadir}{torsion_file}'
        self.SSList = DisulfideList([], 'ALL_PDB_SS')
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []
        self.QUIET = quiet
        
        # create a dataframe with the following columns for the disulfide conformations 
        # extracted from the structure
        # Torsion_DF_Cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 
        # 'chi5', 'energy', 'ca_distance', 'phi_prox', 'psi_prox', 'phi_dist', 'psi_dist']
        
        # SS_df = pd.DataFrame(columns=Torsion_DF_Cols, index=['source'])
        idlist = []

        if subset:
            self.PickleFile = f'{datadir}{SS_SUBSET_PICKLE_FILE}'
            self.PickleDictFile = f'{datadir}{SS_SUBSET_DICT_PICKLE_FILE}'
            self.TorsionFile = f'{datadir}{SS_SUBSET_TORSIONS_FILE}'
        
        if verbose:
            print(f'--> DisulfideLoader(): Reading disulfides from: {self.PickleFile}... ', end='')
        
        with open(self.PickleFile, 'rb') as f:
            sslist = pickle.load(f)
            self.SSList = sslist

        if verbose:
            print(f'done.',)

        self.TotalDisulfides = len(self.SSList)
        
        if verbose:
            print(f'--> DisulfideLoader(): Reading disulfide dict2 from: {self.PickleDictFile}...', end='')
        
        with open(self.PickleDictFile, 'rb') as f:
            self.SSDict = pickle.load(f)
            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)

        if verbose:
            print(f'done.')

        if verbose:
            print(f'--> DisulfideLoader(): Reading Torsion DF from: {self.TorsionFile}...', end='')

        tmpDF  = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()
        if verbose:    
            print(f'Loading complete.')
            self.describe()
        return

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
            indices = self.SSDict[item] # PDB_SS['4yys'] return a list of SS
            res = DisulfideList([], item)
            sslist = self.SSList
            res = DisulfideList([sslist[indices[i]] for i in indices], item)
            # set the resolution for the resulting SS list
            res.resolution = res[0].resolution

        except KeyError:
            try:
                res = self.SSList.get_by_name(item)
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
    
    def average_resolution(self) -> float:
        '''
        Compute and return the average structure resolution for the given list.

        :return: Average resolution (A)
        '''
        res = 0.0
        cnt = 1
        ssdict = self.SSDict

        for ssid,v in ssdict:
            _res = ss.resolution
            if _res is not None and res != -1.0:
                res += _res
                cnt += 1
        return res / cnt

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
    
    def get_by_name(self, name: str) -> Disulfide:
        '''
        Return a Disulfide by its name

        :param name: Disulfide name e.g. '4yys_22A_65A'
        :return: Disulfide

        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import Load_PDB_SS
        >>> from proteusPy.DisulfideList import DisulfideList
            
        Instantiate the Loader with the SS database subset.

        >>> PDB_SS = Load_PDB_SS(verbose=False, subset=True)
        >>> ss1 = PDB_SS.get_by_name('4yys_22A_65A')
        >>> ss1
        <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>
        '''

        _sslist = DisulfideList([], 'tmp')
        _sslist = self.SSList

        res = _sslist.get_by_name(name)

        return res
    
    def describe(self):
        '''
        Provides information about the Disulfide database contained in ```self```.
        '''

        tot = self.TotalDisulfides
        pdbs = len(self.SSDict)
        ram = (sys.getsizeof(self.SSList) + sys.getsizeof(self.SSDict) + sys.getsizeof(self.TorsionDF)) / (1024 * 1024)
        ssMin, ssMax = self.SSList.minmax_energy()

        print(f'========= RCSB Disulfide Database Summary {proteusPy.__version__} =========')
        print(f'PDB IDs present: {pdbs}')
        print(f'Disulfides loaded: {tot}')
        print(f'Total RAM Used by dataset: {ram:.2f} GB.')
        print(f'Best Disulfide: {ssMin.repr_compact()}')
        print(f'Worst Disulfide: {ssMax.repr_compact()}')
       
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
    
    def save(self, savepath=DATA_DIR, verbose=False, subset=False):
        '''
        Save a copy of the fully instantiated Loader to the specified file.

        :param savepath: Path to save the file, defaults to DATA_DIR
        :param fname: Filename, defaults to LOADER_FNAME
        :param verbose: Verbosity, defaults to False
        '''
        if subset:
            fname = LOADER_SUBSET_FNAME
        else:
            fname = LOADER_FNAME

        _fname = f'{savepath}{fname}'

        if verbose:
            print(f'-> DisulfideLoader.save(): Writing {_fname}... ', end='')
        
        with open(_fname, 'wb+') as f:
            pickle.dump(self, f)
        if verbose:
            print(f'done.')
    
    
          
# class ends

def Load_PDB_SS(loadpath=DATA_DIR, verbose=False, subset=False) -> DisulfideLoader:
    '''
    Load the fully instantiated Disulfide database from the specified file. Use the
    defaults unless you are building the database by hand.

    :param load: Path from which to load, defaults to DATA_DIR
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
 
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
