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

pn.extension('mathjax')
pn.extension('vtk', sizing_mode='stretch_width', template='fast')

pn.state.template.param.update(title="RCSB Disulfide Browser")

import pyvista as pv
from pyvista import set_plot_theme

import pyvista as pv
from proteusPy.Disulfide import *

PDB_SS = Load_PDB_SS(verbose=True, subset=False)

_rcsid = '2q7q'
_ssidlist = [
    '2q7q_75D_140D',
    '2q7q_81D_113D',
    '2q7q_88D_171D',
    '2q7q_90D_138D',
    '2q7q_91D_135D',
    '2q7q_98D_129D',
    '2q7q_130D_161D']

RCSB_list = sorted(PDB_SS.IDList)

def click_plot(event):
    create_plot()

# Widgets

rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value="2q7q_75D_140D", options=_ssidlist)

#rcsb_ss_widget.param.watch(get_ss_id, 'value')

button = pn.widgets.Button(name='Render', button_type='primary')
button.on_click(click_plot)

styles_group = pn.widgets.RadioBoxGroup(name='Rending Styles', options=['Split Bonds', 'CPK', 'Ball and Stick'], inline=False)
single_checkbox = pn.widgets.Checkbox(name='Single View', value=False)
rcsb_selector_widget = pn.widgets.Select(name="RCSB ID", value="2q7q", options=RCSB_list)
output_md = pn.pane.Markdown("# Output goes here")

# controls on sidebar
ss_props = pn.WidgetBox('# Disulfide Selection',
    rcsb_selector_widget, rcsb_ss_widget,
).servable(target='sidebar')

ss_styles = pn.WidgetBox('# Rendering Styles',
    styles_group, single_checkbox, button,
).servable(target='sidebar')


# Callbacks
def get_ss_idlist(event) -> list:
    global PDB_SS
    rcs_id = event.new
    sslist = DisulfideList([],'tmp')
    sslist = PDB_SS[rcs_id]

    print(f'RCS: {rcs_id}, |{sslist}|')
    idlist = [ss.name for ss in sslist]
    rcsb_ss_widget.options = idlist

    return idlist

rcsb_selector_widget.param.watch(get_ss_idlist, 'value')

def get_ss(event) -> Disulfide:
    global PDB_SS
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss

def get_ss_id(event):
    rcsb_ss_widget.value = event.new

def create_plot(single=True):
    global PDB_SS

    styles = {"Split Bonds": 'sb', "CPK":'cpk', "Ball and Stick":'bs'}

    ss = Disulfide()
    plotter = pv.Plotter()
    ss_id = rcsb_ss_widget.value
    style = styles[styles_group.value]
    single = single_checkbox.value
    output_md.object = f'--- Rendering: {ss_id} in style |{style}| single: {single}---'

    ss = PDB_SS[ss_id]
    if ss is None:
        print(f'Cannot find ss_id {ss_id}! Returning!')
        output_md.object = f'Cannot find ss_id {ss_id}! Returning!'
        return
    
    plotter = ss.plot(single=single, style=style)
    
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
    vtkpan = pn.pane.VTK(
        plotter.ren_win, margin=0, sizing_mode='stretch_both', orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings, min_height=600
    )
    return vtkpan

vtkpan = create_plot()

pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)

render_win = pn.Column(vtkpan, output_md)
render_win.servable(target='main')

