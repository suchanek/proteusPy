# Disulfide Bond Display Test
# Author: Eric G. Suchanek, PhD.
# Last revision: 2/18/24 -egs-

import pandas as pd
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList

import pyvista as pv
from pyvista import set_plot_theme

TMP = '/tmp/'
def SS_DisplayTest(ss: Disulfide):
    ss.display(style='bs', single=True)
    ss.display(style='cpk', single=True)
    ss.display(style='sb', single=True)
    ss.display(style='pd', single=False)
    ss.screenshot(style='cpk', single=True, fname=f'{TMP}cpk3.png', verbose=True)
    ss.screenshot(style='sb', single=False, fname=f'{TMP}sb3.png', verbose=True)
    print('--> SS_DisplayTest done.')
    return

def SSlist_DisplayTest(sslist):
    sslist.display(style='cpk')
    sslist.display(style='bs')
    sslist.display(style='sb')
    sslist.display(style='pd')
    sslist.display(style='plain')
    sslist.display_overlay(movie=True, fname=f'{TMP}overlay.mp4')
    sslist.display_overlay(movie=False)
    sslist.screenshot(style='sb', fname=f'{TMP}sslist.png')
    print('--> SS_DisplayTest done.')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = Load_PDB_SS(verbose=True, subset=False)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]
    
    SS_DisplayTest(ss)
    
    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    
    ss4yss = DisulfideList([], '4yss')
    ss4yss = PDB_SS['4yys']

    SSlist_DisplayTest(ss4yss)

    sslist = DisulfideList([], 'last12')
    print('Getting last 12')
    
    sslist = PDB_SS[:12]
    SSlist_DisplayTest(sslist)
    
    exit()
