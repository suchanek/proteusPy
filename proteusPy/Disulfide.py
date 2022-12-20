# Implementation for a Disulfide Bond Class object.
# Based on the original C/C++ implementation by Eric G. Suchanek
# Part of the program Proteus, a program for the analysis and modeling of 
# protein structures, with an emphasis on disulfide bonds.
# Author: Eric G. Suchanek, PhD
# Last revision: 12/18/22
# Cα Cβ Sγ

import proteusPy
from proteusPy import *

from .DisulfideExceptions import *
from .DisulfideGlobals import *
from .proteusGlobals import *

from Bio.PDB import Select, Vector, PDBParser
from Bio.PDB.vectors import calc_dihedral
import math

def distance3d(p1: Vector, p2: Vector):
    '''
    Calculate the 3D Euclidean distance for 2 Vector objects
    
    Arguments: p1, p2 Vector objects of dimensionality 3 (3D)
    Returns: Distance
    '''
    _p1 = p1.get_array()
    _p2 = p2.get_array()
    if (len(_p1) != 3 or len(_p2) != 3):
        raise ProteusPyWarning("--> distance3d() requires vectors of length 3!")
    d = math.dist(_p1, _p2)
    return d

def check_chains(pdbid, pdbdir, verbose=True):
    '''Returns True if structure has multiple chains of identical length, False otherwise'''

    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(pdbid, file=f'{pdbdir}pdb{pdbid}.ent')
    ssbond_dict = structure.header['ssbond'] # dictionary of tuples with SSBond prox and distal
    
    if verbose:
        print(f'ssbond dict: {ssbond_dict}')

    same = False
    model = structure[0]
    chainlist = model.get_list()

    if len(chainlist) > 1:
        chain_lens = []
        if verbose:
            print(f'multiple chains. {chainlist}')
        for chain in chainlist:
            chain_length = len(chain.get_list())
            chain_id = chain.get_id()
            if verbose:
                print(f'Chain: {chain_id}, length: {chain_length}')
            chain_lens.append(chain_length)

        if numpy.min(chain_lens) != numpy.max(chain_lens):
            same = False
            if verbose:
                print(f'chain lengths are unequal: {chain_lens}')
        else:
            same = True
            if verbose:
                print(f'Chains are equal length, assuming the same. {chain_lens}')
    return(same)


class DisulfideList(UserList):
    def __init__(self, iterable, id):
        self.pdb_id = id
        super().__init__(self.validate_ss(item) for item in iterable)

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self.data)))
            name = self.data[0].pdb_id
            sublist = [self.data[i] for i in indices]
            return DisulfideList(sublist, name)    
        return UserList.__getitem__(self, item)
    
    def __setitem__(self, index, item):
        self.data[index] = self.validate_ss(item)

    def insert(self, index, item):
        self.data.insert(index, self.validate_ss(item))

    def append(self, item):
        self.data.append(self.validate_ss(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            self.data.extend(other)
        else:
            self.data.extend(self._validate_ss(item) for item in other)
    
    def validate_ss(self, value):
        if isinstance(value, (proteusPy.disulfide.Disulfide)):
            return value
        raise TypeError(f"Disulfide object expected, got {type(value).__name__}")
    
    def set_id(self, value):
        self.pdb_id = value
    
    def get_id(self):
        return self.pdb_id
 
class DisulfideLoader():
    '''
    This class loads .pkl files created from the ExtractDisulfides() routine 
    and initializes itself with their contents. The Disulfide objects are contained
    in a DisulfideList object and Dict. This makes it possible to access the disulfides by
    array index or PDB structure ID.\n

    Example:
        from proteusPy.disulfide import DisulfideList, Disulfide, DisulfideLoader

        SS1 = DisulfideList([],'tmp1')
        SS2 = DisulfideList([],'tmp2')

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
        self.SSList = DisulfideList([], 'ALL_PDB_SS')
        self.SSDict = {}
        self.TorsionDF = pd.DataFrame()
        self.TotalDisulfides = 0
        self.IDList = []

        # create a dataframe with the following columns for the disulfide conformations extracted from the structure
        df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy']
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
 
  
# float init for class 
_FLOAT_INIT = -999.9

# tqdm progress bar width
_PBAR_COLS = 100

class CysSelect(Select):
    def accept_residue(self, residue):
        if residue.get_name() == 'CYS':
            return True
        else:
            return False

def name_to_id(fname: str):
    '''return an entry id for filename pdb1crn.ent -> 1crn'''
    ent = fname[3:-4]
    return ent

def torad(deg):
    return(numpy.radians(deg))

def todeg(rad):
    return(numpy.degrees(rad))

def parse_ssbond_header_rec(ssbond_dict: dict) -> list:
    '''
    Parse the SSBOND dict returned by parse_pdb_header. 
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        ssbond_dict: the input SSBOND dict
    Returns: a list of tuples representing the proximal, distal residue 
             ids for the disulfide.

    '''
    disulfide_list = []
    for ssb in ssbond_dict.items():
        disulfide_list.append(ssb[1])

    return disulfide_list

#
# function reads a comma separated list of PDB IDs and download the corresponding
# .ent files to the PDB_DIR global. 
# Used to download the list of proteins containing at least one SS bond
# with the ID list generated from: http://www.rcsb.org/
#

def DownloadDisulfides(pdb_home=PDB_DIR, model_home=MODEL_DIR, 
                       verbose=False, reset=False) -> None:
    '''
    Function reads a comma separated list of PDB IDs and downloads them
    to the pdb_home path. 

    Used to download the list of proteins containing at least one SS bond
    with the ID list generated from: http://www.rcsb.org/
    '''

    start = time.time()
    donelines = []
    SS_done = []
    ssfile = None
    
    cwd = os.getcwd()
    os.chdir(pdb_home)

    pdblist = PDBList(pdb=pdb_home, verbose=verbose)
    ssfilename = f'{model_home}{SS_ID_FILE}'
    print(ssfilename)
    
    # list of IDs containing >1 SSBond record
    try:
        ssfile = open(ssfilename)
        Line = ssfile.readlines()
    except Exception:
        raise DisulfideIOException(f'Cannot open file: {ssfile}')

    for line in Line:
        entries = line.split(',')

    print(f'Found: {len(entries)} entries')
    completed = {'xxx'} # set to keep track of downloaded

    # file to track already downloaded entries.
    if reset==True:
        completed_file = open(f'{model_home}ss_completed.txt', 'w')
        donelines = []
        SS_DONE = []
    else:
        completed_file = open(f'{model_home}ss_completed.txt', 'w+')
        donelines = completed_file.readlines()

    if len(donelines) > 0:
        for dl in donelines[0]:
            # create a list of pdb id already downloaded
            SS_done = dl.split(',')

    count = len(SS_done) - 1
    completed.update(SS_done) # update the completed set with what's downloaded

    # Loop over all entries, 
    pbar = tqdm(entries, ncols=_PBAR_COLS)
    for entry in pbar:
        pbar.set_postfix({'Entry': entry})
        if entry not in completed:
            if pdblist.retrieve_pdb_file(entry, file_format='pdb', pdir=pdb_home):
                completed.update(entry)
                completed_file.write(f'{entry},')
                count += 1

    completed_file.close()

    end = time.time()
    elapsed = end - start

    print(f'Overall files processed: {count}')
    print(f'Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')
    os.chdir(cwd)
    return

def build_torsion_df(SSList: DisulfideList) -> pd.DataFrame:
    # create a dataframe with the following columns for the disulfide conformations extracted from the structure
    df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy', 'ca_distance']
    SS_df = pd.DataFrame(columns=df_cols)

    pbar = tqdm(SSList, ncols=_PBAR_COLS, miniters=400000)
    for ss in pbar:
        #pbar.set_postfix({'ID': ss.name}) # update the progress bar

        new_row = [ss.pdb_id, ss.name, ss.proximal, ss.distal, ss.chi1, ss.chi2, 
        		ss.chi3, ss.chi4, ss.chi5, ss.energy, ss.ca_distance]
        # add the row to the end of the dataframe
        SS_df.loc[len(SS_df.index)] = new_row.copy() # deep copy
    
    return SS_df.copy()

def ExtractDisulfides(numb=-1, verbose=False, quiet=False, pdbdir=PDB_DIR, 
                        modeldir=MODEL_DIR, picklefile=SS_PICKLE_FILE, 
                        torsionfile=SS_TORSIONS_FILE, problemfile=PROBLEM_ID_FILE,
                        dictfile=SS_DICT_PICKLE_FILE) -> None:
    '''
    This function creates .pkl files needed for the DisulfideLoader class. The Disulfide 
    objects are contained in a DisulfideList object and Dict within these files. 
    In addition, .csv files containing all of the torsions for the disulfides and 
    problem IDs are written.

    Arguments:
        numb:           number of entries to process, defaults to all
        verbose:        more messages
        quiet:          turns of DisulfideConstruction warnings
        pdbdir:         path to PDB files
        modeldir:       path to resulting .pkl files
        picklefile:     name of the disulfide .pkl file
        torsionfile:    name of the disulfide torsion file .csv created
        problemfile:    name of the .csv file containing problem ids
        dictfile:       name of the .pkl file
    
    Example:
        from proteusPy.Disulfide import ExtractDisulfides, DisulfideLoader, DisulfideList

        ExtractDisulfides(numb=500, pdbdir=PDB_DIR, verbose=False, quiet=True)

        SS1 = DisulfideList([],'All_SS')
        SS2 = DisulfideList([], '4yys')

        PDB_SS = DisulfideLoader()
        SS1 = PDB_SS[0]         <-- returns a Disulfide object at index 0
        SS2 = PDB_SS['4yys']    <-- returns a DisulfideList containing all disulfides for 4yys
        SS3 = PDB_SS[:10]       <-- returns a DisulfideList containing the slice
    '''

    entrylist = []
    problem_ids = []
    bad = 0

    # we use the specialized list class DisulfideList to contain our disulfides
    # we'll use a dict to store DisulfideList objects, indexed by the structure ID
    All_ss_dict = {}
    All_ss_list = []

    start = time.time()
    cwd = os.getcwd()

    # Build a list of PDB files in PDB_DIR that are readable. These files were downloaded
    # via the RCSB web query interface for structures containing >= 1 SS Bond.

    os.chdir(pdbdir)

    ss_filelist = glob.glob(f'*.ent')
    tot = len(ss_filelist)

    if verbose:
        print(f'PDB Directory {pdbdir} contains: {tot} files')

    # the filenames are in the form pdb{entry}.ent, I loop through them and extract
    # the PDB ID, with Disulfide.name_to_id(), then add to entrylist.

    for entry in ss_filelist:
        entrylist.append(name_to_id(entry))

    # create a dataframe with the following columns for the disulfide conformations extracted from the structure
    
    df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy', 'ca_distance']
    SS_df = pd.DataFrame(columns=df_cols)

    # define a tqdm progressbar using the fully loaded entrylist list. If numb is passed then
    # only do the last numb entries.
    if numb > 0:
        pbar = tqdm(entrylist[:numb], ncols=_PBAR_COLS)
    else:
        pbar = tqdm(entrylist, ncols=_PBAR_COLS)

    # loop over ss_filelist, create disulfides and initialize them
    for entry in pbar:
        pbar.set_postfix({'ID': entry, 'Bad': bad}) # update the progress bar

        # returns an empty list if none are found.
        sslist = DisulfideList([], entry)
        sslist = load_disulfides_from_id(entry, model_numb=0, verbose=verbose, quiet=quiet, pdb_dir=pdbdir)
        if len(sslist) > 0:
            for ss in sslist:
                All_ss_list.append(ss)
                new_row = [ss.pdb_id, ss.name, ss.proximal, ss.distal, ss.chi1, ss.chi2, ss.chi3, ss.chi4, ss.chi5, ss.energy, ss.ca_distance]
                # add the row to the end of the dataframe
                SS_df.loc[len(SS_df.index)] = new_row.copy() # deep copy
    
            All_ss_dict[entry] = sslist
        else:
            # at this point I really shouldn't have any bad non-parsible file
            bad += 1
            problem_ids.append(entry)
            os.remove(f'pdb{entry}.ent')
    
    if bad > 0:
        if verbose:
            print(f'Found and removed: {len(problem_ids)} problem structures.')
        prob_cols = ['id']
        problem_df = pd.DataFrame(columns=prob_cols)
        problem_df['id'] = problem_ids

        print(f'Saving problem IDs to file: {modeldir}{problemfile}')
        problem_df.to_csv(f'{modeldir}{problemfile}')
    else:
        if verbose:
            print('No problems found.')
   
    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    fname = f'{modeldir}{picklefile}'
    print(f'Saving {len(All_ss_list)} Disulfides to file: {fname}')
    
    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_list, f)

    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    dict_len = len(All_ss_dict)
    fname = f'{modeldir}{dictfile}'

    print(f'Saving {len(All_ss_dict)} Disulfide-containing PDB IDs to file: {fname}')

    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_dict, f)

    fname = f'{modeldir}{torsionfile}'
    if True:
        print(f'Saving torsions to file: {fname}')

    SS_df.to_csv(fname)

    end = time.time()
    elapsed = end - start

    print(f'Disulfide Extraction complete! Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')

    # return to original directory
    os.chdir(cwd)
    return
   

# NB - this only works with the EGS modified version of  BIO.parse_pdb_header.py
def load_disulfides_from_id(struct_name: str, 
                            pdb_dir = '.',
                            model_numb = 0, 
                            verbose = False,
                            quiet=False,
                            dbg = False) -> list:
    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: a list of Disulfide objects initialized from the file.
    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      SS_list = load_disulfides_from_id('5rsa', verbose=True)

      loads the Disulfides from the file and initialize the disulfide objects, returning
      them in the result. '''

    i = 1
    proximal = distal = -1
    SSList = DisulfideList([], struct_name)
    _chaina = None
    _chainb = None

    parser = PDBParser(PERMISSIVE=True)
    
    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[model_numb]

    if verbose:
        print(f'-> load_disulfide_from_id() - Parsing structure: {struct_name}:')

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    # list of tuples with (proximal distal chaina chainb)
    ssbonds = parse_ssbond_header_rec(ssbond_dict) 

    with warnings.catch_warnings():
        if quiet:
            #warnings.filterwarnings("ignore", category=DisulfideConstructionWarning)
            warnings.filterwarnings("ignore")
        for pair in ssbonds:
            # in the form (proximal, distal, chain)
            proximal = pair[0] 
            distal = pair[1]      
            chain1_id = pair[2]
            chain2_id = pair[3]

            if not proximal.isnumeric() or not distal.isnumeric():
                mess = f' -> Cannot parse SSBond record (non-numeric IDs): {struct_name} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring.'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            else:
                proximal = int(proximal)
                distal = int(distal)
            
            if proximal == distal:
                mess = f' -> Cannot parse SSBond record (proximal == distal): {struct_name} Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring.'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue

            _chaina = model[chain1_id]
            _chainb = model[chain2_id]

            if (_chaina is None) or (_chainb is None):
                mess = f' -> NULL chain(s): {struct_name}: {proximal} {chain1_id} - {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue

            if (chain1_id != chain2_id):
                if verbose:
                    mess = (f' -> Cross Chain SS for: Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}')
                    warnings.warn(mess, DisulfideConstructionWarning)
                    pass # was break

            try:
                prox_res = _chaina[proximal]
                dist_res = _chainb[distal]
                
            except KeyError:
                mess = f'Cannot parse SSBond record (KeyError): {struct_name} Prox:  {proximal} {chain1_id} Dist: {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            
            # make a new Disulfide object, name them based on proximal and distal
            # initialize SS bond from the proximal, distal coordinates
            
            if _chaina[proximal].is_disordered() or _chainb[distal].is_disordered():
                mess = f'Disordered chain(s): {struct_name}: {proximal} {chain1_id} - {distal} {chain2_id}, ignoring!'
                warnings.warn(mess, DisulfideConstructionWarning)
                continue
            else:
                if verbose:
                    print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1_id} - {distal} {chain2_id}')
                ssbond_name = f'{struct_name}_{proximal}{chain1_id}_{distal}{chain2_id}'       
                new_ss = Disulfide(ssbond_name)
                new_ss.initialize_disulfide_from_chain(_chaina, _chainb, proximal, distal)
                SSList.append(new_ss)
        i += 1
    return SSList

def check_header_from_file(filename: str,
                            model_numb = 0, 
                            verbose = False,
                            dbg = False) -> bool:

    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: a list of Disulfide objects initialized from the file.
    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      SS_list = load_disulfides_from_id('5rsa', verbose=True)

      loads the Disulfides from the file and initialize the disulfide objects, returning
      them in the result. '''

    i = 1
    proximal = distal = -1
    SSList = []
    _chaina = None
    _chainb = None

    parser = PDBParser(PERMISSIVE=True)
    
    # Biopython uses the Structure -> Model -> Chain hierarchy to organize
    # structures. All are iterable.

    structure = parser.get_structure('tmp', file=filename)
    struct_name = structure.get_id()
    
#    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[model_numb]

    if verbose:
        print(f'-> check_header_from_file() - Parsing file: {filename}:')

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    # list of tuples with (proximal distal chaina chainb)
    ssbonds = parse_ssbond_header_rec(ssbond_dict) 

    for pair in ssbonds:
        # in the form (proximal, distal, chain)

        proximal = pair[0] 
        distal = pair[1]

        if not proximal.isnumeric() or not distal.isnumeric():
            if verbose:
                print(f' ! Cannot parse SSBond record (non-numeric IDs): {struct_name} Prox:  {proximal} {chain1_id} Dist: {distal} {chain2_id}')
            continue # was pass
        else:
            proximal = int(proximal)
            distal = int(distal)
        
        chain1_id = pair[2]
        chain2_id = pair[3]

        _chaina = model[chain1_id]
        _chainb = model[chain2_id]

        if (chain1_id != chain2_id):
            if verbose:
                print(f' -> Cross Chain SS for: Prox: {proximal} {chain1_id} Dist: {distal} {chain2_id}')
                pass # was break

        try:
            prox_res = _chaina[proximal]
            dist_res = _chainb[distal]
        except KeyError:
            print(f' ! Cannot parse SSBond record (KeyError): {struct_name} Prox:  <{proximal}> {chain1_id} Dist: <{distal}> {chain2_id}')
            continue
         
        # make a new Disulfide object, name them based on proximal and distal
        # initialize SS bond from the proximal, distal coordinates
        if (_chaina is not None) and (_chainb is not None):
            if _chaina[proximal].is_disordered() or _chainb[distal].is_disordered():
                continue
            else:
                if verbose:
                   print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1_id} - {distal} {chain2_id}')
        else:
            if dbg:
                print(f' -> NULL chain(s): {struct_name}: {proximal} {chain1_id} - {distal} {chain2_id}')
        i += 1
    return True

def check_header_from_id(struct_name: str, 
                            pdb_dir = '.',
                            model_numb = 0, 
                            verbose = False,
                            dbg = False) -> bool:
    '''
    Loads all Disulfides by PDB ID and initializes the Disulfide objects.
    Assumes the file is downloaded in the pdb_dir path.
    
    NB: Requires EGS-Modified BIO.parse_pdb_header.py 

    Arguments: 
        struct_name: the name of the PDB entry.

        pdb_dir: path to the PDB files, defaults to PDB_DIR

        model_numb: model number to use, defaults to 0 for single
        structure files.

        verbose: print info while parsing

    Returns: True if the proximal and distal residues are CYS and there are no cross-chain SS bonds

    Example:
      Assuming the PDB_DIR has the pdb5rsa.ent file in place calling:

      SS_list = []
      goodfile = check_header_from_id('5rsa', verbose=True)

    '''

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(struct_name, file=f'{pdb_dir}pdb{struct_name}.ent')
    model = structure[0]

    ssbond_dict = structure.header['ssbond'] # NB: this requires the modified code

    bondlist = []
    i = 0

    # get a list of tuples containing the proximal, distal residue IDs for all SSBonds in the chain.
    bondlist = parse_ssbond_header_rec(ssbond_dict)
    
    if len(bondlist) == 0:
        if (verbose):
            print(f'-> check_header_from_id(): no bonds found in bondlist.')
        return False
    
    for pair in bondlist:
        # in the form (proximal, distal, chain)
        proximal = pair[0]
        distal = pair[1]
        chain1 = pair[2]
        chain2 = pair[3]

        chaina = model[chain1]
        chainb = model[chain2]

        try:
            prox_residue = chaina[proximal]
            dist_residue = chainb[distal]
            
            prox_residue.disordered_select("CYS")
            dist_residue.disordered_select("CYS")

            if prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS':
                if (verbose):
                    print(f'build_disulfide() requires CYS at both residues: {prox_residue.get_resname()} {dist_residue.get_resname()}')
                return False
        except KeyError:
            if (dbg):
                print(f'Keyerror: {struct_name}: {proximal} {chain1} - {distal} {chain2}')
                return False
 
        if verbose:
            print(f' -> SSBond: {i}: {struct_name}: {proximal} {chain1} - {distal} {chain2}')

        i += 1
    return True

# Class defination for a structure-based Disulfide Bond.

class Disulfide:
    """
    The Disulfide Bond is characterized by the atomic coordinates N, Cα, Cβ, C', Sγ 
    for both residues, the dihedral angles Χ1 - Χ5 for the disulfide bond conformation,
    a name, proximal resiude number and distal residue number, and conformational energy.
    All atomic coordinates are represented by the BIO.PDB.Vector class.
    """
    def __init__(self, name="SSBOND"):
        """
        Initialize the class. All positions are set to the origin. The optional string name may be passed.
        """
        self.name = name
        self.proximal = -1
        self.distal = -1
        self.energy = _FLOAT_INIT
        self.proximal_chain = str('')
        self.distal_chain = str('')
        self.pdb_id = str('')
        self.proximal_residue_fullid = str('')
        self.distal_residue_fullid = str('')
        self.PERMISSIVE = bool(True)
        self.QUIET = bool(True)
        #
        self.ca_distance = _FLOAT_INIT
        self.torsion_vector = numpy.array((_FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT))

        # global coordinates for the Disulfide, typically as returned from the PDB file
        self.n_prox = Vector(0,0,0)
        self.ca_prox = Vector(0,0,0)
        self.c_prox = Vector(0,0,0)
        self.o_prox = Vector(0,0,0)
        self.cb_prox = Vector(0,0,0)
        self.sg_prox = Vector(0,0,0)
        self.sg_dist = Vector(0,0,0)
        self.cb_dist = Vector(0,0,0)
        self.ca_dist = Vector(0,0,0)
        self.n_dist = Vector(0,0,0)
        self.c_dist = Vector(0,0,0)
        self.o_dist = Vector(0,0,0)

        # local coordinates for the Disulfide, computed using the Turtle3D in 
        # Orientation #1 these are generally private.

        self._n_prox = Vector(0,0,0)
        self._ca_prox = Vector(0,0,0)
        self._c_prox = Vector(0,0,0)
        self._o_prox = Vector(0,0,0)
        self._cb_prox = Vector(0,0,0)
        self._sg_prox = Vector(0,0,0)
        self._sg_dist = Vector(0,0,0)
        self._cb_dist = Vector(0,0,0)
        self._ca_dist = Vector(0,0,0)
        self._n_dist = Vector(0,0,0)
        self._c_dist = Vector(0,0,0)
        self._o_dist = Vector(0,0,0)

        # Dihedral angles for the disulfide bond itself, set to _FLOAT_INIT
        self.chi1 = _FLOAT_INIT
        self.chi2 = _FLOAT_INIT
        self.chi3 = _FLOAT_INIT
        self.chi4 = _FLOAT_INIT
        self.chi5 = _FLOAT_INIT

        # I initialize an array for the torsions which will be used for comparisons
        self.dihedrals = numpy.array((_FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT, _FLOAT_INIT), "d")

    def reset(self):
        self.__init__(self)
    
    # comparison operators, used for sorting. keyed to SS bond energy
    def __lt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy < other.energy

    def __le__(self, other):
        if isinstance(other, Disulfide):
            return self.energy <= other.energy
    
    def __gt__(self, other):
        if isinstance(other, Disulfide):
            return self.energy > other.energy

    def __ge__(self, other):
        if isinstance(other, Disulfide):
            return self.energy >= other.energy

    def __eq__(self, other):
        if isinstance(other, Disulfide):
            return self.energy == other.energy

    def __ne__(self, other):
        if isinstance(other, Disulfide):
            return self.energy != other.energy
    
    # repr functions. The class is large, so I split it up into sections
    def repr_ss_info(self):
        """
        Representation for the Disulfide class
        """
        s1 = f'<Disulfide {self.name} SourceID: {self.pdb_id} Proximal: {self.proximal} {self.proximal_chain} Distal: {self.distal} {self.distal_chain}'
        return s1
    
    def repr_ss_coords(self):
        s2 = f'\nProximal Coordinates:\n   N: {self.n_prox}\n   Cα: {self.ca_prox}\n   C: {self.c_prox}\n   O: {self.o_prox}\n   Cβ: {self.cb_prox}\n   Sγ: {self.sg_prox}\n\n'
        s3 = f'Distal Coordinates:\n   N: {self.n_dist}\n   Cα: {self.ca_dist}\n   C: {self.c_dist}\n   O: {self.o_dist}\n   Cβ: {self.cb_dist}\n   Sγ: {self.sg_dist}\n\n'
        stot = f'{s2} {s3}'
        return stot

    def repr_ss_conformation(self):
        s4 = f'Conformation: (Χ1-Χ5):  {self.chi1:.3f}°, {self.chi2:.3f}°, {self.chi3:.3f}°, {self.chi4:.3f}° {self.chi5:.3f}° '
        s5 = f'Energy: {self.energy:.3f} kcal/mol'
        stot = f'{s4} {s5}'
        return stot

    def repr_ss_local_coords(self):
        """
        Representation for the Disulfide class, internal coordinates.
        """
        s2i = f'Proximal Internal Coordinates:\n   N: {self._n_prox}\n   Cα: {self._ca_prox}\n   C: {self._c_prox}\n   O: {self._o_prox}\n   Cβ: {self._cb_prox}\n   Sγ: {self._sg_prox}\n\n'
        s3i = f'Distal Internal Coordinates:\n   N: {self._n_dist}\n   Cα: {self._ca_dist}\n   C: {self._c_dist}\n   O: {self._o_dist}\n   Cβ: {self._cb_dist}\n   Sγ: {self._sg_dist}\n'
        stot = f'{s2i} {s3i}'
        return stot
    
    def repr_ss_chain_ids(self):
        return(f'Proximal Chain fullID: <{self.proximal_residue_fullid}> Distal Chain fullID: <{self.distal_residue_fullid}>')

    def __repr__(self):
        """
        Representation for the Disulfide class
        """
        
        s1 = self.repr_ss_info()
        res = f'{s1}>'
        return res

    def pprint(self):
        """
        pretty print general info for the Disulfide
        """
        
        s1 = self.repr_ss_info()
        s4 = self.repr_ss_conformation()
        res = f'{s1} {s4}>'
        return res

    def pprint_all(self):
        """
        pretty print all info for a Disulfide
        """
        
        s1 = self.repr_ss_info() + '\n'
        s2 = self.repr_ss_coords()
        s3 = self.repr_ss_local_coords()
        s4 = self.repr_ss_conformation()
        s5 = self.repr_chain_ids()
        res = f'{s1} {s5} {s2} {s3} {s4} >'
        return res

    def _handle_SS_exception(self, message):
        """Handle exception (PRIVATE).

        This method catches an exception that occurs in the Disulfide
        object (if PERMISSIVE), or raises it again, this time adding the
        PDB line number to the error message.
        """
        # message = "%s at line %i." % (message)
        message = f'{message}'

        if self.PERMISSIVE:
            # just print a warning - some residues/atoms may be missing
            warnings.warn(
                "DisulfideConstructionException: %s\n"
                "Exception ignored.\n"
                "Some atoms may be missing in the data structure."
                % message,
                DisulfideConstructionWarning,
            )
        else:
            # exceptions are fatal - raise again with new message (including line nr)
            raise DisulfideConstructionException(message) from None

    def print_compact(self):
        return(f'{self.repr_ss_info()} {self.repr_ss_conformation()}')

    def repr_conformation(self):
        return(f'{self.repr_ss_conformation()}')
    
    def repr_coords(self):
        return(f'{self.repr_ss_coords()}')

    def repr_internal_coords(self):
        return(f'{self.repr_ss_local_coords()}')

    def repr_chain_ids(self):
        return(f'{self.repr_ss_chain_ids()}')

    def set_permissive(self, perm: bool) -> None:
        self.PERMISSIVE = perm
    
    def get_permissive(self) -> bool:
        return self.PERMISIVE

    def set_quiet(self, perm: bool) -> None:
        self.QUIET = perm
    
    def get_quiet(self) -> bool:
        return self.QUIET

    def get_full_id(self):
        return((self.proximal_residue_fullid, self.distal_residue_fullid))
    
    def initialize_disulfide_from_chain(self, chain1, chain2, proximal, distal):
        '''
        Initialize a new Disulfide object with atomic coordinates from the proximal and 
        distal coordinates, typically taken from a PDB file.

        Arguments: 
            chain1: list of Residues in the model, eg: chain = model['A']
            chain2: list of Residues in the model, eg: chain = model['A']
            proximal: proximal residue sequence ID
            distal: distal residue sequence ID
        
        Returns: none. The internal state is modified.
        '''

        id = chain1.get_full_id()[0]

        self.pdb_id = id
        
        # create a new Disulfide object
        chi1 = chi2 = chi3 = chi4 = chi5 = _FLOAT_INIT

        prox = int(proximal)
        dist = int(distal)

        prox_residue = chain1[prox]
        dist_residue = chain2[dist]

        if (prox_residue.get_resname() != 'CYS' or dist_residue.get_resname() != 'CYS'):
            print(f'build_disulfide() requires CYS at both residues: {prox} {prox_residue.get_resname()} {dist} {dist_residue.get_resname()} Chain: {prox_residue.get_segid()}')

        # set the objects proximal and distal values
        self.set_resnum(proximal, distal)

        self.proximal_chain = chain1.get_id()
        self.distal_chain = chain2.get_id()

        self.proximal_residue_fullid = prox_residue.get_full_id()
        self.distal_residue_fullid = dist_residue.get_full_id()


        # grab the coordinates for the proximal and distal residues as vectors so we can do math on them later

        # proximal residue
        if self.QUIET:
            warnings.filterwarnings("ignore", category=DisulfideConstructionWarning)
        try:
            n1 = prox_residue['N'].get_vector()
            ca1 = prox_residue['CA'].get_vector()
            c1 = prox_residue['C'].get_vector()
            o1 = prox_residue['O'].get_vector()
            cb1 = prox_residue['CB'].get_vector()
            sg1 = prox_residue['SG'].get_vector()
            
        except Exception:
            raise DisulfideConstructionWarning(f"Invalid or missing coordinates for proximal residue {proximal}") from None
        
        # distal residue
        try:
            n2 = dist_residue['N'].get_vector()
            ca2 = dist_residue['CA'].get_vector()
            c2 = dist_residue['C'].get_vector()
            o2 = dist_residue['O'].get_vector()
            cb2 = dist_residue['CB'].get_vector()
            sg2 = dist_residue['SG'].get_vector()

        except Exception:
            raise DisulfideConstructionWarning(f"Invalid or missing coordinates for proximal residue {distal}") from None
        
        # update the positions and conformation
        self.set_positions(n1, ca1, c1, o1, cb1, sg1, n2, ca2, c2, o2, cb2, sg2)
        
        # calculate and set the disulfide dihedral angles
        self.chi1 = numpy.degrees(calc_dihedral(n1, ca1, cb1, sg1))
        self.chi2 = numpy.degrees(calc_dihedral(ca1, cb1, sg1, sg2))
        self.chi3 = numpy.degrees(calc_dihedral(cb1, sg1, sg2, cb2))
        self.chi4 = numpy.degrees(calc_dihedral(sg1, sg2, cb2, ca2))
        self.chi5 = numpy.degrees(calc_dihedral(sg2, cb2, ca2, n2))

        self.ca_distance = distance3d(self.ca_prox, self.ca_dist)
        self.torsion_array = numpy.array((self.chi1, self.chi2, self.chi3, self.chi4, self.chi5))

        # calculate and set the SS bond torsional energy
        self.compute_disulfide_torsional_energy()

        # compute and set the local coordinates
        self.compute_local_disulfide_coords()

    def set_chain_id(self, chain_id):
        self.chain_id = chain_id

    def set_positions(self, n_prox: Vector, ca_prox: Vector, c_prox: Vector,
                      o_prox: Vector, cb_prox: Vector, sg_prox: Vector, 
                      n_dist: Vector, ca_dist: Vector, c_dist: Vector,
                      o_dist: Vector, cb_dist: Vector, sg_dist: Vector):
        '''
        Sets the atomic positions for all atoms in the disulfide bond.
        Arguments:
            n_prox
            ca_prox
            c_prox
            o_prox
            cb_prox
            sg_prox
            n_distal
            ca_distal
            c_distal
            o_distal
            cb_distal
            sg_distal
        Returns: None
        '''

        # deep copy
        self.n_prox = n_prox.copy()
        self.ca_prox = ca_prox.copy()
        self.c_prox = c_prox.copy()
        self.o_prox = o_prox.copy()
        self.cb_prox = cb_prox.copy()
        self.sg_prox = sg_prox.copy()
        self.sg_dist = sg_dist.copy()
        self.cb_dist = cb_dist.copy()
        self.ca_dist = ca_dist.copy()
        self.n_dist = n_dist.copy()
        self.c_dist = c_dist.copy()
        self.o_dist = o_dist.copy()

    def set_conformation(self, chi1, chi2, chi3, chi4, chi5):
        '''
        Sets the 5 dihedral angles chi1 - chi5 for the Disulfide object and computes the torsional energy.
        
        Arguments: chi, chi2, chi3, chi4, chi5 - Dihedral angles in degrees (-180 - 180) for the Disulfide conformation.
        Returns: None
        '''

        self.chi1 = chi1
        self.chi2 = chi2
        self.chi3 = chi3
        self.chi4 = chi4
        self.chi5 = chi5
        self.dihedrals = list([chi1, chi2, chi3, chi4, chi5])
        self.compute_disulfide_torsional_energy()

    def set_name(self, namestr="Disulfide"):
        '''
        Sets the Disulfide's name
        Arguments: (str)namestr
        Returns: none
        '''

        self.name = namestr

    def set_resnum(self, proximal, distal):
        '''
        Sets the Proximal and Distal Residue numbers for the Disulfide
        Arguments: 
            Proximal: Proximal residue number
            Distal: Distal residue number
        Returns: None
        '''

        self.proximal = proximal
        self.distal = distal

    def TorsionDistance(p1: Vector, p2: Vector):
        '''
        Calculate the 5D Euclidean distance for 2 Disulfide torsion_vector objects. This is used
        to compare Disulfide Bond torsion angles to determine their torsional 
        'distance'.
        
        Arguments: p1, p2 Vector objects of dimensionality 5 (5D)
        Returns: Distance
        '''
        _p1 = p1.get_array()
        _p2 = p2.get_array()
        if (len(_p1) != 5 or len(_p2) != 5):
            raise ProteusPyWarning("--> distance5d() requires vectors of length 5!")
        d = math.dist(_p1, _p2)
        return d

    def compute_disulfide_torsional_energy(self):
        '''
        Compute the approximate torsional energy for the Disulfide's conformation.
        Arguments: chi1, chi2, chi3, chi4, chi5 - the dihedral angles for the Disulfide
        Returns: Energy (kcal/mol)
        '''
        # @TODO find citation for the ss bond energy calculation
        chi1 = self.chi1
        chi2 = self.chi2
        chi3 = self.chi3
        chi4 = self.chi4
        chi5 = self.chi5

        energy = 2.0 * (cos(torad(3.0 * chi1)) + cos(torad(3.0 * chi5)))
        energy += cos(torad(3.0 * chi2)) + cos(torad(3.0 * chi4))
        energy += 3.5 * cos(torad(2.0 * chi3)) + 0.6 * cos(torad(3.0 * chi3)) + 10.1

        self.energy = energy

    def compute_local_disulfide_coords(self):
        """
        Compute the internal coordinates for a properly initialized Disulfide Object.
        Arguments: SS initialized Disulfide object
        Returns: None, modifies internal state of the input
        """

        turt = Turtle3D('tmp')
        # get the coordinates as numpy.array for Turtle3D use.
        n = self.n_prox.get_array()
        ca = self.ca_prox.get_array()
        c = self.c_prox.get_array()
        cb = self.cb_prox.get_array()
        o = self.o_prox.get_array()
        sg = self.sg_prox.get_array()

        sg2 = self.sg_dist.get_array()
        cb2 = self.cb_dist.get_array()
        ca2 = self.ca_dist.get_array()
        c2 = self.c_dist.get_array()
        n2 = self.n_dist.get_array()
        o2 = self.o_dist.get_array()

        turt.orient_from_backbone(n, ca, c, cb, ORIENT_SIDECHAIN)
        
        # internal (local) coordinates, stored as Vector objects
        # to_local returns numpy.array objects

        self._n_prox = Vector(turt.to_local(n))
        self._ca_prox = Vector(turt.to_local(ca))
        self._c_prox = Vector(turt.to_local(c))
        self._o_prox = Vector(turt.to_local(o))
        self._cb_prox = Vector(turt.to_local(cb))
        self._sg_prox = Vector(turt.to_local(sg))

        self._n_dist = Vector(turt.to_local(n2))
        self._ca_dist = Vector(turt.to_local(ca2))
        self._c_dist = Vector(turt.to_local(c2))
        self._o_dist = Vector(turt.to_local(o2))
        self._cb_dist = Vector(turt.to_local(cb2))
        self._sg_dist = Vector(turt.to_local(sg2))

    def build_disulfide_model(self, turtle: Turtle3D):
        """
        Build a model Disulfide based on the internal dihedral angles.
        Routine assumes turtle is in orientation #1 (at Ca, headed toward
        Cb, with N on left), builds disulfide, and updates the object's internal
        coordinate state. It also adds the distal protein backbone,
        and computes the disulfide conformational energy.

        Arguments: turtle: Turtle3D object properly oriented for the build.
        Returns: None. The Disulfide object's internal state is updated.
        """

        tmp = Turtle3D('tmp')
        tmp.copy_coords(turtle)

        n = Vector(0, 0, 0)
        ca = Vector(0, 0, 0)
        cb = Vector(0, 0, 0)
        c = Vector(0, 0, 0)

        self.ca_prox = tmp._position
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_prox = n
        self.ca_prox = ca
        self.c_prox = c

        tmp.bbone_to_schain()
        tmp.move(1.53)
        tmp.roll(self.chi1)
        tmp.yaw(112.8)
        self.cb_prox = tmp._position

        tmp.move(1.86)
        tmp.roll(self.chi2)
        tmp.yaw(103.8)
        self.sg_prox = tmp._position

        tmp.move(2.044)
        tmp.roll(self.chi3)
        tmp.yaw(103.8)
        self.sg_dist = tmp._position

        tmp.move(1.86)
        tmp.roll(self.chi4)
        tmp.yaw(112.8)
        self.cb_dist = tmp._position

        tmp.move(1.53)
        tmp.roll(self.chi5)
        tmp.pitch(180.0)
        tmp.schain_to_bbone()
        n, ca, cb, c = build_residue(tmp)

        self.n_dist = n
        self.ca_dist = ca
        self.c_dist = c

        self.compute_torsional_energy()

# Class defination ends
 
# End of file
