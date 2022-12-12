# DisulfideExtractor.py
# Author: Eric G. Suchanek, PhD.
# Last modification: 12/3/22 -egs-
#
# Purpose:
# This program processes all the PDB *.ent files in PDB_DIR and creates an array of Disulfide objects representing the Disulfide
# bonds contained in the scanned directory. 
# Outputs are saved in MODEL_DIR:
# 1) SS_PICKLE_FILE: The list of Disulfide objects initialized from the PDB file scan
# 2) SS_TORSIONS_FILE: a .csv containing the SS torsions for the disulfides scanned
# 3) PROBLEM_ID: a .csv containining the problem ids.

import glob
import os
import time
import datetime

from tqdm import tqdm
import pandas as pd
import pickle

from Bio.PDB import PDBParser

# Eric's modules
from ProteusGlobals import *

from Disulfide import DisulfideList, load_disulfides_from_id, name_to_id

def DisulfideExtractor(pdbdir=PDB_DIR, modeldir=MODEL_DIR,
                        picklefile=SS_PICKLE_FILE, torsionsfile=SS_TORSIONS_FILE,
                        problemfile=PROBLEM_ID_FILE, dictfile=SS_DICT_PICKLE_FILE):
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
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
    print(f'PDB Directory {pdbdir} contains: {tot} files')

    # the filenames are in the form pdb{entry}.ent, I loop through them and extract
    # the PDB ID, with Disulfide.name_to_id(), then add to entrylist.

    for entry in ss_filelist:
        entrylist.append(name_to_id(entry))

    # create a dataframe with the following columns for the disulfide conformations extracted from the structure
    df_cols = ['source', 'ss_id', 'proximal', 'distal', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy']
    SS_df = pd.DataFrame(columns=df_cols)

    # define a tqdm progressbar using the fully loaded entrylist list.
    pbar = tqdm(entrylist[:100], ncols=120)

    # loop over ss_filelist, create disulfides and initialize them
    for entry in pbar:
        pbar.set_postfix({'ID': entry, 'Bad': bad}) # update the progress bar

        # returns an empty list if none are found.
        sslist = DisulfideList([], entry)
        sslist = load_disulfides_from_id(entry, model_numb=0, verbose=False, pdb_dir=PDB_DIR)
        if len(sslist) > 0:
            for ss in sslist:
                All_ss_list.append(ss)
                # update the dataframe
                new_row = [entry, ss.name, ss.proximal, ss.distal, ss.chi1, ss.chi2, ss.chi3, ss.chi4, ss.chi5, ss.energy]
                # add the row to the end of the dataframe
                SS_df.loc[len(SS_df.index)] = new_row.copy() # deep copy
            All_ss_dict[entry] = sslist
        else:
            # at this point I really shouldn't have any bad non-parsible file
            bad += 1
            problem_ids.append(entry)
            os.remove(f'pdb{entry}.ent')


    print(f'Found and removed: {len(problem_ids)} problem structures.')
    prob_cols = ['id']
    problem_df = pd.DataFrame(columns=prob_cols)

    problem_df['id'] = problem_ids

    print(f'Saving problem IDs to {modeldir}{problemfile}')
    problem_df.to_csv(f'{modeldir}{problemfile')

    # Save the torsions to a .csv
    print(f'Saving torsions to {modeldir}{torsionfile}')
    SS_df.to_csv(f'{modeldir}{torsionfile}')

    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    fname = f'{modeldir}{sspicklefile}'
    print(f'Saving {len(All_ss_list)} Disulfides to {fname}')
    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_list, f)

    # dump the all_ss array of disulfides to a .pkl file. ~520 MB.
    dict_len = len(All_ss_dict)
    fname = f'{modeldir}{dictfile}'
    print(f'Saving {len(All_ss_dict)} Disulfide-containing PDB IDs to file: {fname}')

    with open(fname, 'wb+') as f:
        pickle.dump(All_ss_dict, f)

    end = time.time()
    elapsed = end - start

    print(f'Complete.\nElapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')


# end of file
