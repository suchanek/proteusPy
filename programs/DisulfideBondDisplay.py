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

PDB_SS = None
PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)

#
#
# split out the left/right handed torsions from the dataframe
#
# retrieve the torsions dataframe

_cols = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5', 'energy', 'ca_distance']

_SSdf = PDB_SS.getTorsions()

# there are a few structures with bad SSBonds. Their
# CA distances are > 7.0. We remove them from consideration
# below

_far = _SSdf['ca_distance'] >= 7.0
_near = _SSdf['ca_distance'] < 7.0
_zed = _SSdf['ca_distance'] == 0.0

SS_df_zed = _SSdf[_zed].copy()
SS_df_Far = _SSdf[_far].copy()
SS_df = _SSdf[_near].copy()

SS_df = SS_df[_cols].copy()

#print(SS_df.head(5))
SS_df.describe()

# make two dataframes containing left handed and right handed disulfides

_left = SS_df['chi3'] <= 0.0
_right = SS_df['chi3'] > 0.0

# left handed and right handed torsion dataframes
SS_df_Left = SS_df[_left]
SS_df_Right = SS_df[_right]

# routine creates 2 lists  for left-handed and right-handed disulfides 
ss_list = PDB_SS.getlist()
left_handed_SS = DisulfideList([], 'left_handed')
right_handed_SS = DisulfideList([], 'right_handed')

i = 0

for i in range(0, len(ss_list)):
    ss = ss_list[i]
    if ss.chi3 < 0:
        left_handed_SS.append(ss)
    else:
        right_handed_SS.append(ss)


print(f'Left Handed: {len(left_handed_SS)}, Right Handed: {len(right_handed_SS)}')


# Set the figure sizes and axis limits.
DPI = 220
WIDTH = 6.0
HEIGHT = 3.0
TORMIN = -179.0
TORMAX = 180.0
GRIDSIZE = 20

import pyvista as pv
from pyvista import Plotter

def main():
    pl = pv.Plotter(window_size=(2048, 2048),)
    pl.enable_anti_aliasing('msaa')

    for i in range(2):
        render_ss(PDB_SS[i], pl)
    
    pl.show()

if __name__ == '__main__':
    main()


