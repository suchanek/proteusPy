'''
DisulfideExtractor.py
Author: Eric G. Suchanek, PhD.
Last modification: 1/24/23 -egs-

Purpose:
This program processes all the PDB *.ent files in PDB_DIR and creates an array of 
Disulfide objects representing the Disulfide bonds contained in the scanned directory. 
Outputs are saved in MODEL_DIR:
- SS_PICKLE_FILE: The list of Disulfide objects initialized from the PDB file scan
- SS_TORSIONS_FILE: a .csv containing the SS torsions for the disulfides scanned
- PROBLEM_ID: a .csv containining the problem ids.

'''

import shutil
from shutil import copytree, ignore_patterns
import time
import datetime

from proteusPy.Disulfide import Extract_Disulfides

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files, created with DisulfideDownloader.py
PDB = '/Users/egs/PDB/good/'
REPO_DATA = '/Users/egs/repos/proteusPy/data/'
MODULE_DATA = '/Users/egs/repos/proteusPy/proteusPy/data/'

# location of the compressed Disulfide .pkl files
DATA = f'{PDB_BASE}data/'

# setting up specific pkl files for a small extraction.
# you must use these exact names, regardless of the number
# extracted in order for the DisulfideLoader class to load them
# correctly. 
#
# don't set these at all for the default total extraction

_SS_PICKLE_FILE = 'PDB_subset_ss.pkl'
_SS_DICT_PICKLE_FILE = 'PDB_subset_ss_dict.pkl'
_SS_TORSIONS_FILE = 'PDB_subset_SS_torsions.csv'
_PROBLEM_ID_FILE = 'PDB_subset_SS_problems.csv'

start = time.time()
Extract_Disulfides(numb=-1, verbose=False, quiet=True, pdbdir=PDB, datadir=DATA)

'''
Extract_Disulfides(numb=1000, pdbdir=PDB, datadir=DATA,
                dictfile=_SS_DICT_PICKLE_FILE,
                picklefile=_SS_PICKLE_FILE,
                torsionfile=_SS_TORSIONS_FILE,
                problemfile=_PROBLEM_ID_FILE,
                verbose=False, quiet=True)
'''

# total extraction uses numb=-1 and takes about 1.5 hours on
# my 2021 MacbookPro M1 Pro computer.
# Extract_Disulfides(numb=-1, pdbdir=PDB, datadir=DATA,
#                 verbose=False, quiet=True)

update = True

if update:
    print(f'Copying: {DATA} to {REPO_DATA}')
    copytree(DATA, REPO_DATA, dirs_exist_ok=True, ignore=ignore_patterns('*_all_*'))

    print(f'Copying: {DATA} to {MODULE_DATA}')
    copytree(DATA, MODULE_DATA, dirs_exist_ok=True, ignore=ignore_patterns('*_all_*'))

end = time.time()

elapsed = end - start
print(f'Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')

# end of file
