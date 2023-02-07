# DisulfideBond Playground
# Playing with the DisulfideBond class
# Author: Eric G. Suchanek, PhD.
# (c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
# License: MIT
# Last Modification: 1/30/23
# Cα Cβ Sγ

import pandas as pd

import pyvista as pv
from pyvista import set_plot_theme

from Bio.PDB import *

# for using from the repo we 
import proteusPy
from proteusPy import *
from proteusPy.data import *
from proteusPy.Disulfide import *

# override any default PDB globals
# location for PDB repository
PDB_ROOT = '/Users/egs/PDB/'

# location of cleaned PDB files - these are not stored in the repo
PDB_GOOD = '/Users/egs/PDB/good/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_ROOT}models/'

# pyvista setup for notebooks
#pv.set_jupyter_backend('ipyvtklink')
#set_plot_theme('dark')

PDB_DIR = '/Users/egs/PDB/good/'

def extract_firstchain_ss(sslist: DisulfideList, verbose=False) -> DisulfideList:
    '''
    Function extracts disulfides from the first chain

    :param sslist: Starting SS list
    :return: SS list from first chain ID or cross-chain
    '''
    id = ''
    chainlist = []
    pc = dc = ''
    res = DisulfideList([], sslist.id)
    xchain = 0

    # build ist of chains
    for ss in sslist:
        pc = ss.proximal_chain
        dc = ss.distal_chain
        if pc != dc:
            xchain += 1
            if verbose:
                print(f'Cross chain ss: {ss}')
        chainlist.append(pc)
    chain = chainlist[0]

    for ss in sslist:
        if ss.proximal_chain == chain:
            res.append(ss)
    
    return res, xchain

def prune_extra_ss(sslist: DisulfideList):
    '''
    Given a dict of disulfides, check for extra chains, grab only the disulfides from
    the first chain and return a dict containing only the first chain disulfides

    :param ssdict: input dictionary with disulfides
    '''
    xchain = 0

    #print(f'Processing: {ss} with: {sslist}')
    id = sslist.pdb_id
    pruned_list = DisulfideList([], id)
    pruned_list, xchain = extract_firstchain_ss(sslist)
        
    return copy.deepcopy(pruned_list), xchain

# Comment these out since they take so long.
# Download_Disulfides(pdb_home=PDB_ORIG, model_home=MODELS, reset=False)

#Extract_Disulfides(numb=1000, pdbdir=PDB_GOOD, datadir=MODELS, verbose=False, quiet=False)

PDB_SS = None
#PDB_SS = DisulfideLoader(verbose=True, picklefile=SS_PICKLE_FILE, pickle_dict_file=SS_DICT_PICKLE_FILE,
#                        torsion_file=SS_TORSIONS_FILE)

PDB_SS = DisulfideLoader(verbose=True, subset=False)

# given the full dictionary, walk through all the keys (PDB ID)
# for each PDB_ID SS list, find and extract the SS for the first chain
# update the 'pruned' dict with the now shorter SS list

_PBAR_COLS = 105
ssdict = {}
ssdict = PDB_SS.SSDict
empty = DisulfideList([], 'empty')

tot = len(ssdict)

# make a dict with an initial bogus value, but properly initialized with an SS list
pruned_dict = {'xxx': empty}

xchain_tot = 0
removed_tot = 0

pbar = tqdm(range(tot), ncols=_PBAR_COLS)

# walk the dict, prune the SS list. This takes > 8 minutes on my Macbook Pro
# for the full dataset.

for _, pdbid_tuple in zip(pbar, enumerate(ssdict)):
    xchain = 0
    removed = 0

    # print(f'{k} {pdbid_tuple}')
    pdbid = pdbid_tuple[1]
    sslist = ssdict[pdbid]
    pruned, xchain = prune_extra_ss(sslist)
    removed = len(sslist) - len(pruned)
    removed_tot += removed
    xchain_tot += xchain
    pruned_dict[pdbid] = pruned
    
print(f'Pruned {removed_tot}, Xchain: {xchain_tot}')

# dump the all_ss array of disulfides to a .pkl file. ~520 MB.
datadir = '/Users/egs/PDB/data/'
picklefile = 'PDB_SS_pruned_dict.pkl'

fname = f'{datadir}{picklefile}'

with open(fname, 'wb+') as f:
    pickle.dump(pruned_dict, f)

# now build the SS list
pruned_list = DisulfideList([], 'PDB_SS_SINGLE_CHAIN')

tot = len(pruned_dict)
pbar = tqdm(range(tot), ncols=_PBAR_COLS)

for _, pdbid_tuple in zip(pbar, enumerate(pruned_dict)):
    # print(f'{k} {pdbid_tuple}')
    pdbid = pdbid_tuple[1]
    sslist = pruned_dict[pdbid]
    pruned_list.extend(sslist)
    
print(f'Total SS: {pruned_list.length}')

# dump the all_ss array of disulfides to a .pkl file. ~520 MB.
datadir = '/Users/egs/PDB/data/'
picklefile = 'PDB_SS_pruned_list.pkl'

fname = f'{datadir}{picklefile}'

with open(fname, 'wb+') as f:
    pickle.dump(pruned_list, f)

# finally build and dump the torsions

datadir = '/Users/egs/PDB/data/'
torsfile = 'PDB_SS_pruned_torsions.csv'
fname = f'{datadir}{torsfile}'

tot = len(pruned_list)

tors_df = pd.DataFrame(columns=Torsion_DF_Cols)
tors_df = pruned_list.build_torsion_df()

tors_df.to_csv(fname)

# end of file


