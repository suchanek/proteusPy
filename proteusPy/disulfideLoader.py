#
# disufideLoader class
# Eric G. Suchanek, PhD. 

import pandas as pd
import pickle
import sys

from proteusPy import *
from proteusPy.disulfide import Disulfide
from proteusPy.disulfide import DisulfideList
from proteusPy.DisulfideExceptions import DisulfideException
from proteusPy.proteusGlobals import MODEL_DIR
from proteusPy.DisulfideGlobals import SS_DICT_PICKLE_FILE, SS_PICKLE_FILE, SS_TORSIONS_FILE, SS_ID_FILE

class DisulfideLoader():
    '''
    This class loads .pkl files created from the DisulfideExtractor() routine 
    and initializes itself with their contents. The Disulfide objects are contained
    in a DisulfideList object and Dict. This makes it possible to access the disulfides by
    array index or PDB structure ID.\n

    Example:
        from Disulfide import DisulfideList, Disulfide, DisulfideLoader

        SS1 = DisulfideList([],'All_SS')
        SS2 = DisulfideList([], '4yys')

        PDB_SS = DisulfideLoader()
        SS1 = PDB_SS[0]         <-- returns a Disulfide object at index 0
        SS2 = PDB_SS['4yys']    <-- returns a DisulfideList containing all disulfides for 4yys
        SS3 = PDB_SS[:10]       <-- returns a DisulfideList containing the slice
    '''

    def __init__(self, verbose=True, modeldir=MODEL_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                torsion_file=SS_TORSIONS_FILE):
        self.ModelDir = modeldir
        self.PickleFile = f'{modeldir}{picklefile}'
        self.PickleDictFile = f'{modeldir}{pickle_dict_file}'
        self.TorsionFile = f'{modeldir}{torsion_file}'
        self.SSList = proteusPy.disulfide.DisulfideList([], 'ALL_PDB_SS')
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []

        # create a dataframe with the following columns for the disulfide conformations extracted from the structure
        df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy']
        SS_df = pd.DataFrame(columns=df_cols, index=['source'])

        idlist = []
        if verbose:
            print(f'Reading disulfides from: {self.PickleFile}')
        
        with open(self.PickleFile, 'rb') as f:
            self.SSList = pickle.load(f)
            
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
        
        #self.TorsionDF = build_torsion_df(self.SSList)
        self.TorsionDF = pd.read_csv(self.TorsionFile)

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

    def getlist(self):
        return self.SSList.copy()
    
    def getdict(self) -> dict:
        return copy.deepcopy(self.SSDict)

    def getTorsions(self):
        return copy.deepcopy(self.TorsionDF)

    def validate_ss(self, value):
        if isinstance(value, (Disulfide)):
            return value
        raise TypeError(
            f"Disulfide object expected, got {type(value).__name__}"
        )
 