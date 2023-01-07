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

def showit(ss: Disulfide):
    ss.display(single=True, style='bs')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]

    ss.display(style='cpk', single=True)
    ss.display(style='sb', single=True)
    ss.display(style='pd', single=True) # fix

    #ss.screenshot(style='cpk', single=True, fname='cpk3.png', verbose=True)
    #ss.screenshot(style='sb', single=False, fname='sb3.png', verbose=True)

    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    
    ss4yss = DisulfideList([], 'tmp')
    ss4yss = PDB_SS['4yys']
    #ss4yss.screenshot(style='sb', fname='ss4yss.png')

    #ss4yss.display('cpk')
    #ss4yss.display('bs')
    #ss4yss.display('sb')
    #ss4yss.display('pd')
    ss4yss.display_overlay()
    
    sslist = DisulfideList([], 'last8')
    sslist = PDB_SS[:8]
    #sslist.screenshot(style='sb', fname='last12.png')

    sslist.display(style='bs')
    sslist.display(style='pd')

    #sslist.display_overlay()

    #ss1j5h = DisulfideList([], 'ss1j5h')
    #ss1j5h = PDB_SS['1j5h']
    #ss1j5h.display(style='sb')
    #ss1j5h.display(style='pd')

    #PDB_SS.display_overlay('1j5h')
    #PDB_SS.display_overlay('4yys')
    
    #ss6fuf = PDB_SS['6fuf']
    #ss6fuf.display(style='bs')

    exit()
