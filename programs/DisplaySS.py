# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/2/23 -egs-
# Cα Cβ Sγ

import pandas as pd
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import DisulfideLoader
from proteusPy.DisulfideList import DisulfideList

import pyvista as pv
from pyvista import set_plot_theme

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files
PDB = '/Users/egs/PDB/good/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_BASE}models/'

import pyvista as pv
from proteusPy.Disulfide import *

def SS_DisplayTest(ss: Disulfide):
    ss.display(style='bs', single=False)
    ss.display(style='cpk')
    ss.display(style='sb', single=True)
    ss.display(style='pd', single=False)
    ss.screenshot(style='cpk', single=True, fname='cpk3.png', verbose=True)
    # ss.screenshot(style='sb', single=False, fname='sb3.png', verbose=True)
    return

def SSlist_DisplayTest(sslist):
    sslist.display(style='cpk')
    sslist.display(style='bs')
    sslist.display(style='sb')
    sslist.display(style='pd')
    sslist.display(style='plain')
    sslist.display_overlay(movie=False, fname='overlay.mp4')
    #sslist.screenshot(style='sb', fname='sslist.png')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, subset=False)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]
    ss.display(single=True)
    ss.display(single=False)
    
    #ss.make_movie(fname='ss_sb.mp4', style='cpk', verbose=True)
    
    #SS_DisplayTest(ss)
    
    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    
    ss4yss = DisulfideList([], '4yss')
    ss4yss = PDB_SS['4yys']

    #SSlist_DisplayTest(ss4yss)

    sslist = DisulfideList([], 'last12')
    print('Getting last 12')
    sslist = PDB_SS[:12]
    
    sslist.display_overlay(movie=False, fname='overlay.mp4')

    #SSlist_DisplayTest(sslist)

    #sslist.display()
    
    #PDB_SS.display_overlay('1j5h')
    #PDB_SS.display_overlay('4yys')
    
    ss6fuf = PDB_SS['6fuf']
    ss6fuf.display(style='sb')

    exit()
