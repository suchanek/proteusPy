from proteusPy import *
from proteusPy.atoms import *
from proteusPy.Disulfide import check_chains
from proteusPy.Disulfide import DisulfideList
from proteusPy.DisulfideExceptions import *

import pyvista as pv

class DisulfideLoader:
    '''
    This class loads .pkl files created from the ExtractDisulfides() routine 
    and initializes itself with their contents. The Disulfide objects are contained
    in a DisulfideList object and Dict. This makes it possible to access the disulfides by
    array index or PDB structure ID. The class can also render Disulfides to a pyVista
    window using the DisulfideLoader.display() method. See below for examples.\n

    Example:
        from proteusPy.Disulfide import DisulfideList, Disulfide, DisulfideLoader

        SS1 = DisulfideList([],'tmp1')
        SS2 = DisulfideList([],'tmp2')

        PDB_SS = DisulfideLoader()
        SS1 = PDB_SS[0]         # returns a Disulfide object at index 0
        SS2 = PDB_SS['4yys']    # returns a DisulfideList containing all disulfides for 4yys
        SS3 = PDB_SS[:10]       # returns a DisulfideList containing the slice

        SSlist = PDB_SS[:8]     # get SS bonds for the last 8 structures
        SSlist.display('sb')    # render the disulfides in 'split bonds' style

    '''

    def __init__(self, verbose=True, modeldir=MODEL_DIR, picklefile=SS_PICKLE_FILE, 
                pickle_dict_file=SS_DICT_PICKLE_FILE,
                torsion_file=SS_TORSIONS_FILE, quiet=True):
        self.ModelDir = modeldir
        self.PickleFile = f'{modeldir}{picklefile}'
        self.PickleDictFile = f'{modeldir}{pickle_dict_file}'
        self.TorsionFile = f'{modeldir}{torsion_file}'
        self.SSList = DisulfideList([], 'ALL_PDB_SS')
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []
        self.QUIET = quiet

        # create a dataframe with the following columns for the disulfide conformations extracted from the structure
        df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy', 'ca_distance', 'phi_prox', 'psi_prox', 'phi_dist', 'psi_dist']
        SS_df = pd.DataFrame(columns=df_cols, index=['source'])
        _SSList = DisulfideList([], 'ALL_PDB_SS')

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

    def getlist(self):
        return self.SSList.copy()
    
    def getdict(self) -> dict:
        return copy.deepcopy(self.SSDict)

    def getTorsions(self, pdbID=None) -> pd.DataFrame:
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
        if isinstance(value, (Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")

    def set_quiet(self, perm: bool) -> None:
        self.QUIET = perm
    
    def get_quiet(self) -> bool:
        return self.QUIET
    
    def copy(self):
        return copy.deepcopy(self)
    
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


    def Odisplay(self, style='bs'):
        ''' 
        Display the Disulfides
        Argument:
            self
        Returns:
            None. Updates internal object.
        '''
        
        ssList = self.SSList
        tot_ss = len(ssList) # number off ssbonds

        cols = 2
        rows = (tot_ss + 1) // cols
        i = 0

        pl = pv.Plotter(window_size=WINSIZE, shape=(rows, cols))
        pl.add_camera_orientation_widget()

        for r in range(rows):
            for c in range(cols):
                pl.subplot(r,c)
                if i < tot_ss:
                    pl.enable_anti_aliasing('msaa')
                    pl.view_isometric()
                    ss = ssList[i]
                    src = ss.pdb_id
                    enrg = ss.energy
                    title = f'{src}: {ss.proximal}{ss.proximal_chain}-{ss.distal}{ss.distal_chain}: {enrg:.2f} kcal/mol'
                    pl.add_title(title=title, font_size=FONTSIZE)
                    pl = ss._render(pl, style=style)
                    near_range, far_range = ss.compute_extents()
                i += 1        
        pl.link_views()
        pl.reset_camera()
        pl.show()

# class ends
