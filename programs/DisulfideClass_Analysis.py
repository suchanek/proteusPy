# DisulfideBond Class Exploration
# Playing with the DisulfideBond class
# Author: Eric G. Suchanek, PhD.
# (c) 2023 Eric G. Suchanek, PhD., All Rights Reserved
# License: MIT
# Last Modification: 2/20/23
# Cα Cβ Sγ

import pandas as pd
import numpy

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

# pyvista setup for notebooks
pv.set_jupyter_backend('trame')
set_plot_theme('dark')

PDB_SS = Load_PDB_SS(verbose=True, subset=False)
_PBAR_COLS = 105

def analyze_classes(loader: DisulfideLoader):
    classes = loader.classdict
    tot_classes = len(classes)

    pbar = enumerate(classes)

    for idx, cls in pbar:
        print(f'{cls} {idx+1}{tot_classes}')

        class_ss_list = loader.from_class(cls)
        class_ss_list.TorsionGraph()

PDB_SS = Load_PDB_SS(verbose=True, subset=False)
analyze_classes(PDB_SS)
