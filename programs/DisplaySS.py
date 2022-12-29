# Disulfide Bond Analysis
# Author: Eric G. Suchanek, PhD.
# Last revision: 12/26/22 -egs-
# Cα Cβ Sγ

import pandas as pd
from proteusPy.disulfide import DisulfideLoader, Disulfide, DisulfideList
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
from proteusPy.disulfide import *

def showit(ss: Disulfide):
    ss.display(single=True, style='bs')

if __name__ == '__main__':
    PDB_SS = None
    PDB_SS = DisulfideLoader(verbose=True, modeldir=MODELS)
    
    # one disulfide from the database
    ss = Disulfide()
    ss = PDB_SS[0]

    ss.display(style='cpk', single=True)
    ss.display(style='cov', single=True)
    ss.display(style='bs', single=False)

    # get all disulfides for one structure. Make a 
    # DisulfideList object to hold it
    '''
    ss4yss = DisulfideList([], 'tmp')
    ss4yss = PDB_SS['4yys']

    ss4yss.display('cpk')
    ss4yss.display('bs')
    ss4yss.display('sb')
    ss4yss.display('plain')
    '''


    sslist = DisulfideList([], 'slice')
    sslist = PDB_SS[:8]
    sslist.display('sb')

    ss1j5h = DisulfideList([], 'ss1j5h')
    ss1j5h = PDB_SS['1j5h']
    ss1j5h.display('sb')

    #PDB_SS.display_overlay('4yys')
    
    #ss.display(single=True, style='sb')

    #pvp = render_disulfides_by_id(PDB_SS, '4yys')
    
    #pvp.show()
    #pvp.close()

    exit()
