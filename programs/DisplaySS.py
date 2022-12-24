# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 12/16/22 -egs-
# Cα Cβ Sγ

import math
import matplotlib
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns

import proteusPy
from proteusPy import *
from proteusPy.disulfide import *
from proteusPy.proteusGlobals import *

import pandas as pd

import pyvista as pv
from pyvista import set_plot_theme

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files
PDB = '/Users/egs/PDB/good/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_BASE}models/'

# when running from the repo the local copy of the Disulfides is in ../pdb/models
# PDB_BASE = '../pdb/'

# location of the compressed Disulfide .pkl files
# MODELS = f'{PDB_BASE}models/'

import pyvista as pv
from pyvista import Plotter

from proteusPy.disulfide import render_Disulfide, render_disulfide_panel, DisulfideLoader

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    ss = PDB_SS[0]

    pvp = render_disulfide_panel(ss)
    pvp.show()
    
    # ss.display(single=True)
    
    exit()


