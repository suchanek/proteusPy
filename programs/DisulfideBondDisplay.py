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

# Set the figure sizes and axis limits.
DPI = 220
WIDTH = 6.0
HEIGHT = 3.0
TORMIN = -179.0
TORMAX = 180.0
GRIDSIZE = 20

import pyvista as pv
from pyvista import Plotter

PDB_SS = None
PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)

if __name__ == '__main__':
    pl = pv.Plotter()
    pl = render_disulfide_panel(PDB_SS[0])
    pl.show()
    exit()


