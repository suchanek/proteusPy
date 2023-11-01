# RCSB Disulfide Bond Database Browser
# Author: Eric G. Suchanek, PhD.
# Last revision: 11/1/23 -egs-
# Cα Cβ Sγ

import pandas as pd
import proteusPy
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import DisulfideLoader, Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList
import panel as pn

pn.extension('mathjax')
pn.extension('vtk', sizing_mode='stretch_width', template='fast')

pn.state.template.param.update(title="RCSB Disulfide Browser v0.1")

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
    global render_win
    plotter = render_ss()
    vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings, min_height=600
    )
    render_win[1] = vtkpan

# Widgets

rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value="2q7q_75D_140D", options=_ssidlist)

button = pn.widgets.Button(name='Render', button_type='primary')
button.on_click(click_plot)

styles_group = pn.widgets.RadioBoxGroup(name='Rending Styles', options=['Split Bonds', 'CPK', 'Ball and Stick'], inline=False)
single_checkbox = pn.widgets.Checkbox(name='Single View', value=True)
shadows_checkbox = pn.widgets.Checkbox(name='Shadows', value=False)
rcsb_selector_widget = pn.widgets.Select(name="RCSB ID", value="2q7q", options=RCSB_list)
title_md = pn.pane.Markdown("# Title")
output_md = pn.pane.Markdown("# Output goes here")
info_md = pn.pane.Markdown("# SS Info")

# controls on sidebar
ss_props = pn.WidgetBox('# Disulfide Selection',
                        rcsb_selector_widget, rcsb_ss_widget
                        ).servable(target='sidebar')

ss_styles = pn.WidgetBox('# Rendering Styles',
                         styles_group, single_checkbox, shadows_checkbox, button, info_md,
                        ).servable(target='sidebar')

# Callbacks
def get_ss_idlist(event):
    global PDB_SS

    rcs_id = event.new
    sslist = DisulfideList([],'tmp')
    sslist = PDB_SS[rcs_id]

    print(f'RCS: {rcs_id}, |{sslist}|')
    idlist = [ss.name for ss in sslist]
    rcsb_ss_widget.options = idlist
    return idlist

rcsb_selector_widget.param.watch(get_ss_idlist, 'value')

def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f'# Disulfide: {name}'
    title_md.object = title

def update_info(ss):
    src = ss.pdb_id
    enrg = ss.energy
    name = ss.name

    info = f'### Disulfide: {name}  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°'
    info_md.object = info

def update_output(output_str):
    output_md.object = output_str

def get_ss(event) -> Disulfide:
    global PDB_SS
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss

def get_ss_id(event):
    rcsb_ss_widget.value = event.new

def render_ss():
    global PDB_SS

    styles = {"Split Bonds": 'sb', "CPK":'cpk', "Ball and Stick":'bs'}

    ss = Disulfide()
    plotter = pv.Plotter()
    ss_id = rcsb_ss_widget.value
    
    ss = PDB_SS[ss_id]
    if ss is None:
        output_md.object = f'Cannot find ss_id {ss_id}! Returning!'
        return

    style = styles[styles_group.value]
    single = single_checkbox.value
    shadows = shadows_checkbox.value
    outputstr = f'--- Rendering: {ss_id} in style |{style}| single: {single}  and shadows {shadows} ---'
    
    update_title(ss)
    update_info(ss)
    update_output(outputstr)

    plotter = ss.plot(single=single, style=style, shadows=shadows)
    
    # save initial camera properties
    renderer = list(plotter.ren_win.GetRenderers())[0]
    initial_camera = renderer.GetActiveCamera()
    
    initial_camera_pos = {
        "focalPoint": initial_camera.GetFocalPoint(),
        "position": initial_camera.GetPosition(),
        "viewUp": initial_camera.GetViewUp()
    }
    return plotter

# Panel creation using the VTK Scene created by the plotter pyvista
orientation_widget = True
enable_keybindings = True
plotter = render_ss()

vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings, min_height=600
    )
pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)

render_win = pn.Column(title_md, vtkpan, output_md)
render_win.servable(target='main')
