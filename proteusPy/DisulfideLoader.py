#
# DisulfideLoader class definition
# This class embodies functions used to load Disulfide Bonds extracted from the
# RCSB Protein Databank and initialize the DisulfideLoader with the Disulfide
# lists and dictionary.
# Author: Eric G. Suchanek, PhD.
# Last modification: 1/22/23

import sys
import copy

import pandas as pd
import pyvista as pv
import pickle

import proteusPy
from proteusPy.ProteusGlobals import *
from proteusPy.atoms import *
from proteusPy.data import *

from proteusPy.Disulfide import Check_chains
from proteusPy.Disulfide import DisulfideList, Torsion_DF_Cols
from proteusPy.DisulfideExceptions import *
from proteusPy.DisulfideGlobals import *

class DisulfideLoader:
    '''
    This class loads .pkl files created from the proteusPy.Disulfide.Extract_Disulfides() routine 
    and initializes itself with their contents. The Disulfide objects are contained
    in a DisulfideList object and Dict, and their torsions and distances stored in a .csv file.
    This makes it possible to access the disulfides by array index or PDB structure ID. 
    The class can also render Disulfides to a pyVista window using the 
    DisulfideLoader.display() method. See below for examples.\n

    Example:
        >>> from proteusPy.Disulfide import Disulfide
        >>> from proteusPy.DisulfideList import DisulfideList
        >>> from proteusPy.DisulfideLoader import DisulfideLoader

        >>> SS1 = DisulfideList([],'tmp1')
        >>> SS2 = DisulfideList([],'tmp2')
        >>> PDB_SS = DisulfideLoader(verbose=False, subset=True)
        >>> SS1 = PDB_SS[0]
        >>> SS1
        <Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>
        >>> SS2 = PDB_SS['4yys']
        >>> SS2
        [<Disulfide 4yys_22A_65A SourceID: 4yys Proximal: 22 A Distal: 65 A>, <Disulfide 4yys_56A_98A SourceID: 4yys Proximal: 56 A Distal: 98 A>, <Disulfide 4yys_156A_207A SourceID: 4yys Proximal: 156 A Distal: 207 A>, <Disulfide 4yys_22B_65B SourceID: 4yys Proximal: 22 B Distal: 65 B>, <Disulfide 4yys_56B_98B SourceID: 4yys Proximal: 56 B Distal: 98 B>, <Disulfide 4yys_156B_207B SourceID: 4yys Proximal: 156 B Distal: 207 B>]
        >>> SSlist = PDB_SS[:4]
        >>> SSlist.display(style='sb') 
    '''

    def __init__(self, verbose=True, datadir=DATA_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                torsion_file=SS_TORSIONS_FILE, quiet=True, subset=False):
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
        '''
        Initialize the class.
        '''
        # create a dataframe with the following columns for the disulfide conformations extracted from the structure
        # Torsion_DF_Cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 
        # 'chi5', 'energy', 'ca_distance', 'phi_prox', 'psi_prox', 'phi_dist', 'psi_dist']
        
        SS_df = pd.DataFrame(columns=Torsion_DF_Cols, index=['source'])
        _SSList = DisulfideList([], 'ALL_PDB_SS')
        idlist = []

        if subset:
            self.PickleFile = f'{datadir}PDB_subset_ss.pkl'
            self.PickleDictFile = f'{datadir}PDB_subset_ss_dict.pkl'
            self.TorsionFile = f'{datadir}PDB_subset_SS_torsions.csv'
        
        if verbose:
            print(f'Reading disulfides from: {self.PickleFile}')
        
        with open(self.PickleFile, 'rb') as f:
            sslist = pickle.load(f)
            self.SSList = sslist

        self.TotalDisulfides = len(self.SSList)
        
        if verbose:
            print(f'Disulfides Read: {self.TotalDisulfides}')
            print(f'Reading disulfide dict from: {self.PickleDictFile}')
        
        with open(self.PickleDictFile, 'rb') as f:
            self.SSDict = pickle.load(f)
            for key in self.SSDict:
                idlist.append(key)
            self.IDList = idlist.copy()
            totalSS_dict = len(self.IDList)
        
        if verbose:
            print(f'Reading Torsion DF {self.TorsionFile}.')
        
        #tmpDF  = pd.read_csv(self.TorsionFile, index_col='source')
        tmpDF  = pd.read_csv(self.TorsionFile)
        tmpDF.drop(tmpDF.columns[[0]], axis=1, inplace=True)

        self.TorsionDF = tmpDF.copy()

        if verbose:
            print(f'Read torsions DF.')
            print(f'PDB IDs parsed: {totalSS_dict}')
            print(f'Total Space Used: {sys.getsizeof(self.SSList) + sys.getsizeof(self.SSDict) + sys.getsizeof(self.TorsionDF)} bytes.')
        return

    # overload __getitem__ to handle slicing and indexing
    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.SSList)))
            # return [self.SSList[i] for i in indices]
            name = self.SSList[0].pdb_id
            sublist = [self.SSList[i] for i in indices]
            return DisulfideList(sublist, name)
        
        if isinstance(item, int):
            if (item < 0 or item >= self.TotalDisulfides):
                mess = f'DisulfideDataLoader error. Index {item} out of range 0-{self.TotalDisulfides - 1}'
                raise DisulfideException(mess)
            else:
                return self.SSList[item]

        try:
            res = self.SSDict[item]
        except KeyError:
            mess = f'! Cannot find key {item} in SSBond dict!'
            raise DisulfideException(mess)
        return res
    
    def __setitem__(self, index, item):
        self.SSList[index] = self.validate_ss(item)

    def getlist(self) -> DisulfideList:
        '''
        Return the list of Disulfides contained in the class.

        :return: Disulfide list
        :rtype: DisulfideList
        '''
        return copy.deepcopy(self.SSList)
    
    def getdict(self) -> dict:
        '''
        Return the Disulfide Dict contained in the class.

        :return: Disulfide Dict
        :rtype: dict
        '''
        return copy.deepcopy(self.SSDict)

    def getTorsions(self, pdbID=None) -> pd.DataFrame:
        '''
        Return the torsions, distances and energies defined by Disulfide.Torsion_DF_cols

        :param pdbID: pdbID, defaults to None, meaning return entire dataset.
        :type pdbID: str, optional
        :raises DisulfideParseWarning: Raised if not found
        :return: Torsions Dataframe
        :rtype: pd.DataFrame
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
    
    def validate_ss(self, value):
        if isinstance(value, (proteusPy.Disulfide.Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")

    def set_quiet(self, perm: bool) -> None:
        self.QUIET = perm
    
    def get_quiet(self) -> bool:
        return self.QUIET
    
    def copy(self):
        return copy.deepcopy(self)

    def get_by_name(self, name):
        _sslist = DisulfideList([], 'tmp')
        _sslist = self.SSList

        res = _sslist.get_by_name(name)

        return res
    
    def display_overlay(self, pdbid: str):
        ''' 
        Render all disulfides for a given PDB ID overlaid in stick mode against
        a common coordinate frames. This allows us to see all of the disulfides
        at one time in a single view. Colors vary smoothy between bonds.
        
        Arguments:
            PDB_SS: DisulfideLoader object initialized with the database.
            pdbid: the actual PDB id string

        Returns: None.    
        ''' 

        ssbonds = self[pdbid]
        ssbonds.display_overlay()

    def display(self, index, style='bs'):
        ''' 
        Display the Disulfides
        Argument:
            self
        Returns:
            None. Updates internal object.
        '''
        
        ssList = self.SSList[index]
        ssList.display(style=style)

# class ends

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# end of file
