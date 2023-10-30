# RCSB Disulfide Bond Database Browser
# Author: Eric G. Suchanek, PhD.
# Last revision: 1/2/23 -egs-
# Cα Cβ Sγ

import pandas as pd
import proteusPy
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import DisulfideLoader, Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList
import panel as pn

pn.extension('vtk', sizing_mode='stretch_width', template='fast')
pn.state.template.param.update(title="RCSB Disulfide Browser")

import pyvista as pv
from pyvista import set_plot_theme

# the locations below represent the actual location on the dev drive.
# location for PDB repository
PDB_BASE = '/Users/egs/PDB/'

# location of cleaned PDB files
PDB = '/Users/egs/PDB/good/'

# location of the compressed Disulfide .pkl files
MODELS = f'{PDB_BASE}models/'

global _rcsid
global _ssidlist

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

def create_plot(rcs_id="2q7q", ss_id='2q7q_75D_140D', single=True):
    global PDB_SS
    ss = Disulfide()
    plotter = pv.Plotter()

    _rcsid = rcs_id
    _ssidlist = [ss.name for ss in PDB_SS[rcs_id]]

    ss = PDB_SS[ss_id]
    plotter = ss.plot(single=single)
    
    # save initial camera properties
    renderer = list(plotter.ren_win.GetRenderers())[0]
    initial_camera = renderer.GetActiveCamera()
    initial_camera_pos = {
        "focalPoint": initial_camera.GetFocalPoint(),
        "position": initial_camera.GetPosition(),
        "viewUp": initial_camera.GetViewUp()
    }

    # Panel creation using the VTK Scene created by the plotter pyvista
    orientation_widget = True
    enable_keybindings = True
    return pn.pane.VTK(
        plotter.ren_win, margin=0, sizing_mode='stretch_both', orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings, min_height=600
    )


_rcsid = '2q7q'
_ssidlist = '2q7q_75D_140D'

PDB_SS = None
PDB_SS = Load_PDB_SS(verbose=True, subset=False)


# one disulfide from the database
ss = Disulfide()
ss = PDB_SS[0]
rcsb_list = sorted(PDB_SS.IDList)
ss_list = [ss.name for ss in PDB_SS['2q7q']]

rcsb_selector_widget = pn.widgets.Select(name="RCSB ID", value="2q7q", options=rcsb_list)
rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value="2q7q_75D_140D", options=ss_list)
bound_plot = pn.bind(create_plot, rcs_id=rcsb_selector_widget, ss_id=rcsb_ss_widget)

interface = pn.Column(rcsb_selector_widget, rcsb_ss_widget)
dashboard = pn.Column(bound_plot)


ss_app = pn.Row(interface, bound_plot)
ss_app.servable()
