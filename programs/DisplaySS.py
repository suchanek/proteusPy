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

def render_all_disulfides(ssList: DisulfideList) -> pv.Plotter:
    ''' 
        Create a pyvista Plotter object with linked four windows for CPK, ball and stick,
        wireframe and surface displays for the Disulfide.
        Argument:
            self
        Returns:
            None. Updates internal object.
        '''
    
    # fontsize
    _fs = 12
    name = ssList.pdb_id
    tot_ss = len(ssList) # number off ssbonds
    title = f'Disulfides from {name}: ({tot_ss} total)'
    cols = 2
    rows = tot_ss // cols

    pl = pv.Plotter(window_size=(1200, 1200), shape=(rows, cols))
    
    i = 0

    for r in range(rows):
        for c in range(cols):
            ss = ssList[i]
            pl.enable_anti_aliasing('msaa')
            pl.view_isometric()
            pl.add_camera_orientation_widget()
            pl.subplot(r,c)
            pl.add_axes()
            pl.add_title(title=title, font_size=_fs)
            pl = render_disulfide(ss, pl, style='bs')
            i += 1
    pl.link_views()
    pl.camera_position = [(0, 0, -20), (0, -2, 0), (0, 1, 0)]
    pl.camera.zoom(.75)
    return pl

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

    tot_ss = len(ss4yss) # number off ssbonds
    print(f'tot {tot_ss}')

    pvp = pv.Plotter()
    pvp = render_all_disulfides(ss4yss)

    # showit(ss)
    #pvp = render_ssbonds_by_id(PDB_SS, '4yys')
    
    pvp.show()

    exit()


