'''
DisulfideExtractor.py

Purpose:
This program encapsulates the steps needed to extract disulfides from the PDB file repository,
build the DisulfideLoader object, and save it into the proteusPy module data directory.

Author: Eric G. Suchanek, PhD.
Last revision: 2/12/23 -egs-
'''

import argparse

from shutil import copytree, ignore_patterns
import time
import datetime

from proteusPy.Disulfide import Extract_Disulfides
from proteusPy.DisulfideLoader import DisulfideLoader

from proteusPy.data import DATA_DIR
from proteusPy.data import SS_PROBLEM_SUBSET_ID_FILE, SS_SUBSET_DICT_PICKLE_FILE
from proteusPy.data import SS_SUBSET_PICKLE_FILE, SS_SUBSET_TORSIONS_FILE

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files, created with DisulfideDownloader.py
PDB_DIR = '/Users/egs/PDB/good/'
MODULE_DATA = '/Users/egs/repos/proteusPy/proteusPy/data/'

# location of the compressed Disulfide .pkl files
DATA_DIR = f'{PDB_BASE}data/'

# The following performs an extraction of 1000 SS and saves them with
# the correct filenames to be read as 'subset'. Do not change the filenames
# defined above

def do_extract(verbose, full, subset, cutoff):
    if subset:
        if verbose:
            print('--> Extracting the SS subset...')
        Extract_Disulfides(
                        numb=1000, 
                        pdbdir=PDB_DIR, 
                        datadir=DATA_DIR,
                        dictfile=SS_SUBSET_DICT_PICKLE_FILE,
                        picklefile=SS_SUBSET_PICKLE_FILE,
                        torsionfile=SS_SUBSET_TORSIONS_FILE,
                        problemfile=SS_PROBLEM_SUBSET_ID_FILE,
                        verbose=False, 
                        quiet=True,
                        dist_cutoff=cutoff
                        )

    # total extraction uses numb=-1 and takes about 1.5 hours on
    # my 2021 MacbookPro M1 Pro computer.

    if full:
        if verbose:
            print('--> Extracting the SS full dataset. This will take ~1.5 hours.')
        Extract_Disulfides(
                        numb=-1, 
                        verbose=False, 
                        quiet=True, 
                        pdbdir=PDB_DIR, 
                        datadir=DATA_DIR, 
                        dist_cutoff=cutoff
                        )
    return

def do_build(verbose, full, subset):
    '''
    Loads and saves a ```proteusPy.DisulfideLoader.DisulfideLoader``` object
    to a .pkl file.

    :param verbose: Verbosity, boolean
    :param full: Whether to load and save the full dataset, boolean
    :param subset: Whether to load and save the subset database, boolean
    '''
    if full:
        if verbose:
            print('--> Building the packed loader for the full dataset...')
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=False)
        PDB_SS.save(savepath=DATA_DIR, subset=False, verbose=verbose)

    if subset:
        if verbose:
            print('--> Building the packed loader for the Disulfide subset...')
        PDB_SS = DisulfideLoader(datadir=DATA_DIR, subset=True)
        PDB_SS.save(savepath=DATA_DIR, subset=True, verbose=verbose)
    
    return

###
def do_stuff(all=False, extract=False, build=True, full=False, update=True, subset=True, verbose=True, cutoff=-1.0):
    '''
    Main entrypoint for the proteusPy Disulfide database extraction and creation workflow.

    :param all: _description_, defaults to False
    :param extract: _description_, defaults to False
    :param build: _description_, defaults to True
    :param full: _description_, defaults to False
    :param update: _description_, defaults to True
    :param subset: _description_, defaults to True
    :param verbose: _description_, defaults to True
    :param cutoff: _description_, defaults to -1.0
    '''
    _extract = extract
    _build = build
    _full = full
    _update = update
    _subset = subset
    _verbose = verbose
    
    if all:
        _extract = _build = _update = _subset = _full = True

    if _extract == True:
        print(f'Extracting...')
        do_extract(_verbose, _full, _subset, cutoff)

    if _build == True:
        print(f'Building...')
        do_build(_verbose, _full, _subset)

    if _update == True:
        print(f'Copying: {DATA_DIR} to {MODULE_DATA}')
        copytree(DATA_DIR, MODULE_DATA, dirs_exist_ok=True, ignore=ignore_patterns('*_pruned_*'))

    return

start = time.time()

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--all", help="do everything. Extract, build and save both datasets", action=argparse.BooleanOptionalAction)
parser.add_argument("-c", "--cutoff", help="distance cutoff for disulfide distance pruning", type=float, required=False)
parser.add_argument("-u", "--update", help="update the repo package", action=argparse.BooleanOptionalAction)
parser.add_argument("-v", "--verbose", help="level of verbosity", action=argparse.BooleanOptionalAction)
parser.add_argument("-e", "--extract", help="extract disulfides from the PDB structure files", action=argparse.BooleanOptionalAction)
parser.add_argument("-f", "--full", help="extract all disulfides from the PDB structure files", action=argparse.BooleanOptionalAction)
parser.add_argument("-b", "--build", help="rebuild the loader", action=argparse.BooleanOptionalAction)
parser.add_argument("-s", "--subset", help="rebuild the subset only", action=argparse.BooleanOptionalAction)

parser.set_defaults(all=False)
parser.set_defaults(update=True)
parser.set_defaults(verbose=True)
parser.set_defaults(extract=True)
parser.set_defaults(subset=True)
parser.set_defaults(build=True)
parser.set_defaults(full=False)
parser.set_defaults(cutoff=-1.0)

args = parser.parse_args()

all = args.all
extract = args.extract
build = args.build
update = args.update
full = args.full
subset = args.subset
verbose = args.verbose
cutoff = args.cutoff

do_stuff(all=all, extract=extract, build=build, full=full, update=update, subset=subset, verbose=verbose, cutoff=cutoff)

end = time.time()
elapsed = end - start

print(f'DisulfideExtractor Complete! Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')

# end of file
