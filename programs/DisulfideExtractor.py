# DisulfideExtractor.py
# Author: Eric G. Suchanek, PhD.
# Last modification: 12/3/22 -egs-
#
# Purpose:
# This program processes all the PDB *.ent files in PDB_DIR and creates an array of 
# Disulfide objects representing the Disulfide bonds contained in the scanned directory. 
# Outputs are saved in MODEL_DIR:
# 1) SS_PICKLE_FILE: The list of Disulfide objects initialized from the PDB file scan
# 2) SS_TORSIONS_FILE: a .csv containing the SS torsions for the disulfides scanned
# 3) PROBLEM_ID: a .csv containining the problem ids.
#

import shutil

from proteusPy.Disulfide import Extract_Disulfides

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files
PDB = '/Users/egs/PDB/good/'
REPO_MODELS = '/Users/egs/repos/proteusPy/proteusPy/data/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_BASE}models/'

# setting up specific pkl files for a small extraction. 
# don't set these at all for the default extraction

_SS_PICKLE_FILE = 'PDB_all_ss.pkl'
_SS_DICT_PICKLE_FILE = 'PDB_all_ss_dict.pkl'
_SS_TORSIONS_FILE = 'PDB_all_SS_torsions.csv'
_PROBLEM_ID_FILE = 'PDB_all_SS_problems.csv'


Extract_Disulfides(numb=1000, pdbdir=PDB, datadir=MODELS,
                dictfile=_SS_DICT_PICKLE_FILE,
                picklefile=_SS_PICKLE_FILE,
                torsionfile=_SS_TORSIONS_FILE,
                problemfile=_PROBLEM_ID_FILE,
                verbose=False, quiet=True)


# Extract_Disulfides(numb=1000, pdbdir=PDB, datadir=MODELS,
#                 verbose=False, quiet=True)

update = True

if update:
    print(f'Copying: {MODELS} to {REPO_MODELS}')
    shutil.copytree(MODELS, REPO_MODELS, dirs_exist_ok=True)

# end of file
