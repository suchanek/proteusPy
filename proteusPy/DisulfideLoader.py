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

from proteusPy.DisulfideList import DisulfideList
from proteusPy.Disulfide import Torsion_DF_Cols
from proteusPy.Disulfide import Disulfide

from proteusPy.DisulfideExceptions import *
from proteusPy.data import *

class DisulfideLoader:
    '''
    This class loads files created from the proteusPy.Disulfide.Extract_Disulfides() routine 
    and initializes itself with their contents. The Disulfide objects are contained
    in a proteuPy.DisulfideList.DisulfideList object and Dict, and their torsions and distances stored in a .csv file.
    This makes it possible to access the disulfides by array index or PDB structure ID. 
    The class can also render Disulfides to a pyVista window using the 
    DisulfideLoader.display() method. See below for examples.\n

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
    >>> SS2 = PDB_SS['4yys']
    >>> SS2
    [<Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56A_98A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156A_207A, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_22B_65B, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_56B_98B, Source: 4yys, Resolution: 1.35 Å>, <Disulfide 4yys_156B_207B, Source: 4yys, Resolution: 1.35 Å>]
    >>> SSlist = PDB_SS[:4]
    >>> SSlist.display(style='sb') 
    '''

    SS_DICT_PICKLE_FILE2 = f'{SS_DICT_PICKLE_FILE}_ind'

    def __init__(self, verbose=True, datadir=DATA_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                pickle_dict_file2=SS_DICT_PICKLE_FILE2,
                torsion_file=SS_TORSIONS_FILE, quiet=True, subset=False):
        '''
        Initializing the class initiates loading either the entire Disulfide dataset,
        or the 'subset', which consists of the first 5000 PDB structures. The subset
        is useful for testing and debugging since it doesn't require nearly as much
        memory or time. The name for the subset file is hard-coded. One can pass a
        different data directory and file names for the pickle files. These different
        directories would be established with the proteusPy.Disulfide.Extract_Disulfides function.
        function.
        '''

        self.ModelDir = datadir
        self.PickleFile = f'{datadir}{picklefile}'
        self.PickleDictFile = f'{datadir}{pickle_dict_file}'
        self.PickleDictFile2 = f'{datadir}{pickle_dict_file2}'
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
        
        SS_df = pd.DataFrame(columns=Torsion_DF_Cols, index=['source'])
        _SSList = DisulfideList([], 'ALL_PDB_SS')

        idlist = []

        if subset:
            self.PickleFile = f'{datadir}PDB_subset_ss.pkl'
            self.PickleDictFile = f'{datadir}PDB_subset_ss_dict.pkl'
            self.PickleDictFile2 = f'{datadir}PDB_subset_ss_dict_ind.pkl'
            self.TorsionFile = f'{datadir}PDB_subset_SS_torsions.csv'
        
        if verbose:
            print(f'--> DisulfideLoader(): Reading disulfides from: {self.PickleFile}')
        
        with open(self.PickleFile, 'rb') as f:
            sslist = pickle.load(f)
            self.SSList = sslist

        self.TotalDisulfides = len(self.SSList)
        '''
        if verbose:
            print(f'--> DisulfideLoader(): Reading disulfide dict from: {self.PickleDictFile}')
        
        with open(self.PickleDictFile, 'rb') as f:
            self.SSDict = pickle.load(f)
            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)
        '''
        if verbose:
            print(f'--> DisulfideLoader(): Reading disulfide dict2 from: {self.PickleDictFile2}')
        
        with open(self.PickleDictFile2, 'rb') as f:
            self.SSDict2 = pickle.load(f)
            for key in self.SSDict2:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)

        if verbose:
            print(f'--> DisulfideLoader(): Reading Torsion DF from: {self.TorsionFile}.')

        tmpDF  = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()
        if verbose:    
            print(f'Loading complete.\nSummary: \n PDB IDs parsed: {totalSS_dict}')
            print(f' Disulfides loaded: {self.TotalDisulfides}')
            print(f' Total RAM Used by dataset: {((sys.getsizeof(self.SSList) + sys.getsizeof(self.SSDict) + sys.getsizeof(self.TorsionDF)) / (1024 * 1024)):.2f} GB.')
        return


    # overload __getitem__ to handle slicing and indexing
    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.SSList)))
            # return [self.SSList[i] for i in indices]
            name = self.SSList[0].pdb_id
            resolution = self.SSList[0].resolution
            sublist = [self.SSList[i] for i in indices]
            return DisulfideList(sublist, name, resolution)
        
        if isinstance(item, int):
            if (item < 0 or item >= self.TotalDisulfides):
                mess = f'DisulfideDataLoader error. Index {item} out of range 0-{self.TotalDisulfides - 1}'
                raise DisulfideException(mess)
            else:
                return self.SSList[item]

        try:
            indices = self.SSDict2[item]
            res = DisulfideList([],'tmp')
            sslist = self.SSList

            tot = len(indices)
            # res = DisulfideList([sslist[indices[i]] for i in range(tot)], 'tmp')
            res = DisulfideList([sslist[indices[i]] for i in indices], 'tmp')


        except KeyError:
            mess = f'! Cannot find key {item} in SSBond dict!'
            raise DisulfideException(mess)
        return res
    
    def __setitem__(self, index, item):
        self.SSList[index] = self._validate_ss(item)

    def getlist(self) -> DisulfideList:
        '''
        Return the list of Disulfides contained in the class.

        :return: Disulfide list
        :rtype: DisulfideList
        '''
        return copy.deepcopy(self.SSList)
    
    def getdict(self, id=None) -> DisulfideList:
        '''
        Return the Disulfide List contained in the ID.
        :param id: PDB ID, e.g. '4yys'
        :return: Disulfide List
        '''
        if id is not None:
            try:
                sslist = self.SSDict[id]
                return copy.deepcopy(sslist)
            except:
                mess = f'Invalid index {id}'
                raise(ProteusPyWarning(mess))
                return {}
        else:
            return copy.deepcopy(self.SSDict)

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
                  proximal       distal         chi1         chi2         chi3  ...     phi_prox     psi_prox     phi_dist     psi_dist  torsion_length
        count  8210.000000  8210.000000  8210.000000  8210.000000  8210.000000  ...  8210.000000  8210.000000  8210.000000  8210.000000     8210.000000
        mean    231.242875   280.309257   -46.635049    -4.363584    -4.438116  ...  -101.610350    69.281822   -98.354988    72.681350      233.892187
        std     285.911354   284.999227   103.687677   110.144551    94.110252  ...    43.440895    95.468934    42.211145    95.374444       56.415414
        min       1.000000     6.000000  -179.990815  -179.990782  -179.502078  ...  -180.000000  -180.000000  -180.000000  -180.000000      102.031030
        25%      44.000000    96.000000   -91.432474   -86.794595   -88.431544  ...  -131.259716   -18.032679  -124.017966   -14.776966      187.399815
        50%     138.000000   195.000000   -64.205495   -54.482183   -64.843442  ...  -105.365621   115.617182  -100.506678   122.338903      232.813196
        75%     324.000000   364.000000   -41.621931   100.943649    93.532489  ...   -73.165302   140.628913   -73.916673   144.177030      273.123655
        max    5045.000000  5070.000000   179.945970   179.987671   179.976934  ...   179.578735   179.972228   179.337160   179.971149      368.022621
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
    
    def _validate_ss(self, value):
        if isinstance(value, (Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")
    
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
    
    def copy(self):
        '''
        Return a copy of self

        :return: Copy of self
        :rtype: proteusPy.DisulfideLoader
        '''
        return copy.deepcopy(self)

    def get_by_name(self, name: str) -> Disulfide:
        '''
        Return a Disulfide by its name

        :param name: Disulfide name e.g. '4yys_22A_65A'
        :return: Disulfide

        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideLoader import DisulfideLoader
        >>> from proteusPy.DisulfideList import DisulfideList
            
        Instantiate the Loader with the SS database subset.

        >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
        >>> ss1 = PDB_SS.get_by_name('4yys_22A_65A')
        >>> ss1
        <Disulfide 4yys_22A_65A, Source: 4yys, Resolution: 1.35 Å>
        '''

        _sslist = DisulfideList([], 'tmp')
        _sslist = self.SSList

        res = _sslist.get_by_name(name)

        return res
    
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
       
# class ends

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
