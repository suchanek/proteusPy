'''
Disulfide class analysis using `proteusPy.Disulfide`. Disulfide families are defined
using the +/- formalism of Hogg et al. (Biochem, 2006, 45, 7429-7433), across
all 32 possible classes ($$2^5$$). Classes are named per Hogg's convention.


+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| IDX|   chi1_s |   chi2_s |   chi3_s |   chi4_s |   chi5_s |   class_id | SS_Classname   | FXN        |
+====+==========+==========+==========+==========+==========+============+================+============+
|  0 |       -1 |       -1 |       -1 |       -1 |       -1 |      00000 | -LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  1 |       -1 |       -1 |       -1 |       -1 |        1 |      00002 | 00002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  2 |       -1 |       -1 |       -1 |        1 |       -1 |      00020 | -LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  3 |       -1 |       -1 |       -1 |        1 |        1 |      00022 | 00022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  4 |       -1 |       -1 |        1 |       -1 |       -1 |      00200 | -RHStaple      | Allosteric |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  5 |       -1 |       -1 |        1 |       -1 |        1 |      00202 | 00202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  6 |       -1 |       -1 |        1 |        1 |       -1 |      00220 | 00220          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  7 |       -1 |       -1 |        1 |        1 |        1 |      00222 | 00222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  8 |       -1 |        1 |       -1 |       -1 |       -1 |      02000 | 02000          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
|  9 |       -1 |        1 |       -1 |       -1 |        1 |      02002 | 02002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 10 |       -1 |        1 |       -1 |        1 |       -1 |      02020 | -LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 11 |       -1 |        1 |       -1 |        1 |        1 |      02022 | 02022          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 12 |       -1 |        1 |        1 |       -1 |       -1 |      02200 | -RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 13 |       -1 |        1 |        1 |       -1 |        1 |      02202 | 02202          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 14 |       -1 |        1 |        1 |        1 |       -1 |      02220 | -RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 15 |       -1 |        1 |        1 |        1 |        1 |      02222 | 02222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 16 |        1 |       -1 |       -1 |       -1 |       -1 |      20000 | ±LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 17 |        1 |       -1 |       -1 |       -1 |        1 |      20002 | +LHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 18 |        1 |       -1 |       -1 |        1 |       -1 |      20020 | ±LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 19 |        1 |       -1 |       -1 |        1 |        1 |      20022 | +LHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 20 |        1 |       -1 |        1 |       -1 |       -1 |      20200 | ±RHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 21 |        1 |       -1 |        1 |       -1 |        1 |      20202 | +RHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 22 |        1 |       -1 |        1 |        1 |       -1 |      20220 | ±RHHook        | Catalytic  |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 23 |        1 |       -1 |        1 |        1 |        1 |      20222 | 20222          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 24 |        1 |        1 |       -1 |       -1 |       -1 |      22000 | -/+LHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 25 |        1 |        1 |       -1 |       -1 |        1 |      22002 | 22002          | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 26 |        1 |        1 |       -1 |        1 |       -1 |      22020 | +/-LHStaple    | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 27 |        1 |        1 |       -1 |        1 |        1 |      22022 | +LHStaple      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 28 |        1 |        1 |        1 |       -1 |       -1 |      22200 | -/+RHHook      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 29 |        1 |        1 |        1 |       -1 |        1 |      22202 | +RHHook        | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 30 |        1 |        1 |        1 |        1 |       -1 |      22220 | ±RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
| 31 |        1 |        1 |        1 |        1 |        1 |      22222 | +RHSpiral      | UNK        |
+----+----------+----------+----------+----------+----------+------------+----------------+------------+
'''


# DisulfideBond Class Exploration
# Author: Eric G. Suchanek, PhD.
# (c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
# License: MIT

# Last Modification: 2/24/2023
# Cα Cβ Sγ

import pandas as pd
import time

import pyvista as pv
from pyvista import set_plot_theme

from Bio.PDB import *

# for using from the repo we 
import proteusPy
from proteusPy import *
from proteusPy.data import *
from proteusPy.Disulfide import *
from proteusPy.DisulfideList import DisulfideList, load_disulfides_from_id
from proteusPy.DisulfideLoader import Load_PDB_SS, DisulfideLoader
from proteusPy.ProteusGlobals import PBAR_COLS

start = time.time()

# pyvista setup for notebooks
pv.set_jupyter_backend('trame')
set_plot_theme('dark')

def analyze_classes(loader: DisulfideLoader, do_graph=True, do_consensus=True) -> DisulfideList:
    class_filename = f'{DATA_DIR}SS_consensus_class32.pkl'
    classes = loader.classdict
    tot_classes = len(classes)
    res_list = DisulfideList([], 'SS_Class_Avg_SS')

    pbar = enumerate(classes)
    for idx, cls in pbar:
        fname = f'{DATA_DIR}classes/ss_class_{idx}.png'
        print(f'--> analyze_classes(): {cls} {idx+1}/{tot_classes}')

        # get the classes
        class_ss_list = loader.from_class(cls)
        if do_graph:
            class_ss_list.display_torsion_statistics(display=False, save=True, 
                fname=fname, light=True, stats=False)

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            print(f'--> analyze_classes(): Computing avg conformation for: {cls}')
            avg_conformation = class_ss_list.Average_Conformation

            # build the average disulfide for the class
            ssname = f'{cls}_avg'
            exemplar = Disulfide(ssname)
            exemplar.build_model(avg_conformation[0], avg_conformation[1],
                                avg_conformation[2],avg_conformation[3],
                                avg_conformation[4])
            res_list.append(exemplar)
        
    if do_consensus:
        print(f'--> analyze_classes(): Writing consensus structures to: {class_filename}')
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)
    
    return res_list

from proteusPy.DisulfideClasses import create_six_class_df
from proteusPy.utility import sort_by_column

def analyze_six_classes(loader: DisulfideLoader, do_graph=True, 
                        do_consensus=True, cutoff=0.1) -> DisulfideList:
    _PBAR_COLS = 85

    class_filename = f'{DATA_DIR}SS_consensus_class_quat.pkl'

    tors = loader.getTorsions()
    six = create_six_class_df(tors)
    tot_classes = six.shape[0]
    res_list = DisulfideList([], 'SS_Class_Avg_SS')
    total_ss = len(loader.SSList)

    pbar = tqdm(range(tot_classes), ncols=_PBAR_COLS)

    # loop over all rows
    for idx in pbar:
        row = six.iloc[idx]
        cls = row['class_id']
        ss_list = row['ss_id']
        tot = len(ss_list)
        if 100 * tot / total_ss <  cutoff:
            continue

        fname = f'{DATA_DIR}classes/ss_class_sext_{cls}.png'
        pbar.set_postfix({'CLS': cls, 'Cnt': tot}) # update the progress bar

        class_disulfides = DisulfideList([], cls, quiet=True)

        pbar2 = tqdm(ss_list, ncols=_PBAR_COLS, leave=False)
        for ssid in pbar2:
            class_disulfides.append(loader[ssid])

        if do_graph:
            class_disulfides.display_torsion_statistics(display=False, save=True, 
                fname=fname, light=True, stats=False)

        if do_consensus:
            # get the average conformation - array of dihedrals
            avg_conformation = np.zeros(5)

            #print(f'--> analyze_six_classes(): Computing avg conformation for: {cls}')
            avg_conformation = class_disulfides.Average_Conformation

            # build the average disulfide for the class
            ssname = f'{cls}_avg'
            exemplar = Disulfide(ssname)
            exemplar.build_model(avg_conformation[0], avg_conformation[1],
                                 avg_conformation[2], avg_conformation[3],
                                 avg_conformation[4])
            res_list.append(exemplar)
        
    if do_consensus:
        print(f'--> analyze_six_classes(): Writing consensus structures to: {class_filename}')
        with open(class_filename, "wb+") as f:
            pickle.dump(res_list, f)
    
    return res_list

# main program begins
PDB_SS = Load_PDB_SS(verbose=True, subset=False)

ss_classlist = DisulfideList([], 'PDB_SS_SIX_CLASSES')
ss_classlist = analyze_six_classes(PDB_SS, do_graph=True, 
                                   do_consensus=True, cutoff=0.5)

end = time.time()
elapsed = end - start

print(f'Disulfide Class Analysis Complete! \nElapsed time: {datetime.timedelta(seconds=elapsed)} (h:m:s)')

# end of file

