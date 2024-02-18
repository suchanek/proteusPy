'''
This module, ``Test_DisplaySS.py``, is part of the proteusPy package, a Python package for 
the analysis and modeling of protein structures, with an emphasis on disulfide bonds. Note:
depending on the speed of your hardware it can some time to load the full database and render
the disulfides.

Author: Eric G. Suchanek, PhD
Last revision: 2/17/2024
'''

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
    print('--> SSList_DisplayTest done.')

def main():
    '''
    Program tests the proteusPy Disulfide rendering routines.
    Usage: run the program and close each window after manipulating it. The program
    will run through display styles and save a few files to /tmp. There should be no
    errors upon execution.
    '''
    
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

    # grab the last 12 disulfides
    sslist = DisulfideList([], 'last12')
    print('Getting last 12')
    
    sslist = PDB_SS[:12]
    SSlist_DisplayTest(sslist)

    print('--> Program complete!')
    
if __name__ == '__main__':
    main()

