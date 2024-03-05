"""
DisulfidePruner.py - program reads the .pkl files created with the ```DisulfideExtractor``` class, extracts the
disulfides from the first chain only, and writes the subset of disulfides to *_subset.pkl files.

Author: Eric G. Suchanek, PhD.
(c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
License: MIT
Last Modification: 2/9/23
"""

# Cα Cβ Sγ

import pandas as pd
import pyvista as pv
from Bio.PDB import *
from pyvista import set_plot_theme

# for using from the repo we
import proteusPy
from proteusPy import *
from proteusPy.data import *
from proteusPy.Disulfide import *
from proteusPy.utility import extract_firstchain_ss, prune_extra_ss

# override the default location for the stored disulfides, which defaults to DATA_DIR
datadir = "/Users/egs/PDB/data/"

_PBAR_COLS = 90
PDB_SS = None

empty = DisulfideList([], "empty")
pruned_dict = {"xxx": empty}
ssdict = {}

xchain_tot = 0
removed_tot = 0

start = time.time()

PDB_SS = DisulfideLoader(verbose=True, subset=False, datadir=datadir)

ssdict = PDB_SS.SSDict
tot = len(ssdict)

# make a dict with an initial bogus value, but properly initialized with an SS list


# walk the dict, prune the SS list. This takes > 8 minutes on my Macbook Pro
# for the full dataset.

print(f"Pruning...")

pbar = tqdm(range(tot), ncols=_PBAR_COLS)

for _, pdbid_tuple in zip(pbar, enumerate(ssdict)):
    xchain = 0
    removed = 0
    pdbid = pdbid_tuple[1]
    pbar.set_postfix(
        {"ID": pdbid, "Rem": removed_tot, "XC": xchain_tot}
    )  # update the progress bar
    sslist = PDB_SS[pdbid]
    pruned, xchain = prune_extra_ss(sslist)
    removed = len(sslist) - len(pruned)
    removed_tot += removed
    xchain_tot += xchain
    pruned_dict[pdbid] = pruned

print(f"Pruned {removed_tot}, Xchain: {xchain_tot}")

# now build the SS list
pruned_list = DisulfideList([], "PDB_SS_SINGLE_CHAIN")

tot = len(pruned_dict)
pbar = tqdm(range(tot), ncols=_PBAR_COLS)

print(f"Building SS list...")

for _, pdbid_tuple in zip(pbar, enumerate(pruned_dict)):
    # print(f'{k} {pdbid_tuple}')
    pdbid = pdbid_tuple[1]
    sslist = pruned_dict[pdbid]
    pruned_list.extend(sslist)

print(f"Total SS: {pruned_list.length}")

# dump the list of disulfides to a .pkl file. ~520 MB.
picklefile = "PDB_pruned_ss.pkl"
fname = f"{datadir}{picklefile}"
print(f"Writing: {fname}")
with open(fname, "wb+") as f:
    pickle.dump(pruned_list, f)

# build the dict from the pruned list
print(f"Building SS dict...")
pruned_dict_ind = {"xxx": []}

tot = pruned_list.length
pbar = tqdm(range(tot), ncols=_PBAR_COLS)

for ss, i in zip(pruned_list, pbar):
    try:
        ss_ind = pruned_dict_ind.pop(ss.pdb_id)
        ss_ind.append(i)
        pruned_dict[ss.pdb_id] = ss_ind
    except KeyError:
        pruned_dict[ss.pdb_id] = [i]

pruned_dict.pop("xxx")

# dump the dict of disulfides to a .pkl file. ~520 MB.
picklefile = "PDB_pruned_ss_dict.pkl"
fname = f"{datadir}{picklefile}"

print(f"Writing: {fname}")

with open(fname, "wb+") as f:
    pickle.dump(pruned_dict, f)

# finally build and dump the torsions

torsfile = "PDB_pruned_ss_torsions.csv"
fname = f"{datadir}{torsfile}"
tot = len(pruned_list)

print(f"Building torsion DF")

tors_df = pd.DataFrame(columns=Torsion_DF_Cols)
tors_df = pruned_list.build_torsion_df()

print(f"Writing: {fname}")
tors_df.to_csv(fname)

end = time.time()

elapsed = end - start
print(f"Complete. Elapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)")

exit()

# end of file
