# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/2/23 -egs-
# Cα Cβ Sγ

import pandas as pd
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import DisulfideLoader

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
from proteusPy.Disulfide import *

def SS_DisplayTest(ss: Disulfide):
    ss.display(style='bs', single='False')
    ss.display(style='cpk')
    ss.display(style='sb', single=True)
    ss.display(style='pd', single=False)
    ss.screenshot(style='cpk', single=True, fname='cpk3.png', verbose=True)
    ss.screenshot(style='sb', single=False, fname='sb3.png', verbose=True)
    return

def SSlist_DisplayTest(sslist):
    sslist.display(style='cpk')
    sslist.display(style='bs')
    sslist.display(style='sb')
    sslist.display(style='pd')
    sslist.display_overlay()
    sslist.screenshot(style='pd', fname='sslist.png')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]
    #SS_DisplayTest(ss)
    

    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    
    ss4yss = DisulfideList([], '4yss')
    ss4yss = PDB_SS['4yys']

    #SSlist_DisplayTest(ss4yss)

    sslist = DisulfideList([], 'last8')
    sslist = PDB_SS[:12]
    sslist.display_overlay()

    #SSlist_DisplayTest(sslist)

    #sslist.display()
    
    #PDB_SS.display_overlay('1j5h')
    #PDB_SS.display_overlay('4yys')
    
    ss6fuf = PDB_SS['6fuf']
    ss6fuf.display(style='sb')

    exit()
