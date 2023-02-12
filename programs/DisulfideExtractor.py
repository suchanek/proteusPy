'''
DisulfideExtractor.py
Author: Eric G. Suchanek, PhD.
Last modification: 2/11/23 -egs-

Purpose:
This program processes all the PDB *.ent files in PDB_DIR and creates an array of 
Disulfide objects representing the Disulfide bonds contained in the scanned directory. 
Outputs are saved in MODEL_DIR:
- SS_PICKLE_FILE: The list of Disulfide objects initialized from the PDB file scan
- SS_TORSIONS_FILE: a .csv containing the SS torsions for the disulfides scanned
- PROBLEM_ID: a .csv containining the problem ids.
'''

from shutil import copytree, ignore_patterns
import time
import datetime

from proteusPy.Disulfide import Extract_Disulfides

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files, created with DisulfideDownloader.py
PDB_DIR = '/Users/egs/PDB/good/'
REPO_DATA = '/Users/egs/repos/proteusPy/data/'
MODULE_DATA = '/Users/egs/repos/proteusPy/proteusPy/data/'

# location of the compressed Disulfide .pkl files
DATA_DIR = f'{PDB_BASE}data/'

# Setting up specific pkl files for a small extraction.
# you must use these exact names, regardless of the number
# extracted in order for the DisulfideLoader class to load them
# correctly. 
#
# Don't set these at all for the default total extraction
#

from proteusPy.data import SS_PROBLEM_SUBSET_ID_FILE, SS_SUBSET_DICT_PICKLE_FILE
from proteusPy.data import SS_SUBSET_PICKLE_FILE, SS_SUBSET_TORSIONS_FILE

start = time.time()

# The following performs an extraction of 1000 SS and saves them with
# the correct filenames to be read as 'subset'. Do not change the filenames
# defined above

Extract_Disulfides(
                numb=1000, 
                pdbdir=PDB_DIR, 
                datadir=DATA_DIR,
                dictfile=SS_SUBSET_DICT_PICKLE_FILE,
                picklefile=SS_SUBSET_PICKLE_FILE,
                torsionfile=SS_SUBSET_TORSIONS_FILE,
                problemfile=SS_PROBLEM_SUBSET_ID_FILE,
                verbose=False, quiet=True,
                dist_cutoff=-1.0
                )

# total extraction uses numb=-1 and takes about 1.5 hours on
# my 2021 MacbookPro M1 Pro computer.

Extract_Disulfides(
                numb=-1, 
                verbose=False, 
                quiet=True, 
                pdbdir=PDB_DIR, 
                datadir=DATA_DIR, 
                dist_cutoff=-1.0
                )

update = True

if update:
    print(f'Copying: {DATA_DIR} to {MODULE_DATA}')
    copytree(DATA_DIR, MODULE_DATA, dirs_exist_ok=True, ignore=ignore_patterns('*_pruned_*'))
    #copytree(DATA_DIR, MODULE_DATA, dirs_exist_ok=True)

end = time.time()

elapsed = end - start
print(f'Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')
exit()

# end of file
