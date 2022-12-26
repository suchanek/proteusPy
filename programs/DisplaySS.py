# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 12/16/22 -egs-
# Cα Cβ Sγ

import pandas as pd

import proteusPy
from proteusPy import *
from proteusPy.disulfide import *
from proteusPy.proteusGlobals import *

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

from proteusPy.disulfide import render_disulfide
from proteusPy.disulfide import DisulfideLoader

def showit(ss: Disulfide):
    ss.display(single=False, style='cpk')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]
    print(f'SS: {ss}')

    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    ss4yss = DisulfideList([], '4yys')
    ss4yss = PDB_SS['4yys']

    ss4crn = DisulfideList([], '1crn')
    ss4crn = PDB_SS['1crn']

    tot_ss = len(ss4yss) # number off ssbonds
    print(f'tot {tot_ss}')

    pvp = pv.Plotter()
    #pvp = render_all_disulfides(ss4yss)
    pvp = render_disulfide_panel(ss4yss[0])
    # showit(ss)
    #pvp = render_disulfides_by_id(PDB_SS, '4yys')
    
    pvp.show()

    exit()


