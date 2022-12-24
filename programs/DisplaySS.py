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

    
def render_ssbonds_by_id(PDB_SS, pdbid: str) -> pv.Plotter():
    ''' 
    Given a pv.Plotter() object (usually bunch of Disulfides), put a window and axes
    around it and return the plotter object    
    '''
    
    _fs = 12 # fontsize

    ss = PDB_SS[pdbid]
    tot_ss = len(ss) # number off ssbonds
    title = f'Disulfides for {pdbid} ({tot_ss} total)'

    pl = pv.Plotter(window_size=(1200, 1200))
    
    pl.add_title(title=title, font_size=_fs)

    pl.enable_anti_aliasing('msaa')
    pl.view_isometric()
    pl.add_camera_orientation_widget()
    pl.add_axes()

    pl.camera_position = [(0, 0, -20), (0, -2, 0), (0, 1, 0)]
    pl.camera.zoom(.75)
    
    # make a colormap in vector space
    # starting and ending colors
    strtc = numpy.array([0, .1, 0])
    endc = numpy.array([1, .2, .0])
    vlen = math.dist(strtc, endc)

    cdir = numpy.array([0,0,0])
    newcol = numpy.array([0,0,0])

    # color direction vector and length
    cdir = endc - strtc
    clen = math.dist(strtc, endc)
    cdir /= clen

    # delta along color vector
    dlta = clen / tot_ss

    print(f'Cdir: {cdir}')

    i = 0
    for ssbond in ss:
        print(f'SS: {ssbond}')
        mycolor = strtc + cdir * i * dlta
        pl = render_disulfide(ssbond, pl, style='st', bondcolor=mycolor)
        i += 1
    return pl

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    ss = PDB_SS[0]  
    pvp = pv.Plotter()

    print(f'SS: {ss}')
    # showit(ss)
    pvp = render_ssbonds_by_id(PDB_SS, '4yys')
    pvp.show()


    exit()


