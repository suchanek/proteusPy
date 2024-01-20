'''
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 11/2/2023
'''

import sys
import time
import pyvista as pv
import panel as pn

import proteusPy

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList

pn.extension('vtk', sizing_mode='stretch_width', template='fast')

_vers = 0.5

_rcsid = '2q7q'
_default_ss = '2q7q_75D_140D'
_ssidlist = [
    '2q7q_75D_140D',
    '2q7q_81D_113D',
    '2q7q_88D_171D',
    '2q7q_90D_138D',
    '2q7q_91D_135D',
    '2q7q_98D_129D',
    '2q7q_130D_161D']

PDB_SS = Load_PDB_SS(verbose=True, subset=False)

vers = PDB_SS.version
tot = PDB_SS.TotalDisulfides
pdbs = len(PDB_SS.SSDict)
orientation_widget = True
enable_keybindings = True

RCSB_list = sorted(PDB_SS.IDList)

pn.state.template.param.update(title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}")

def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "default"


def click_plot(event):
    """Force a re-render of the currently selected disulfide. Removes the pane
    and re-adds it to the panel.

    Returns:
        None
    """
    global render_win, vtkpan
    
    plotter = render_ss()
    vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', 
                     orientation_widget=orientation_widget,
                     enable_keybindings=enable_keybindings, min_height=500)
    
    vtkpan.param.trigger('object')
    render_win[0] = vtkpan
    
    # this position is dependent on the vtk panel position in the render_win pane!
    print(f'RenderWin: {render_win}')

# Widgets

rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value=_default_ss, options=_ssidlist)

button = pn.widgets.Button(name='Refresh', button_type='primary')
button.on_click(click_plot)

styles_group = pn.widgets.RadioBoxGroup(name='Rending Style', 
                                        options=['Split Bonds', 'CPK', 'Ball and Stick'], 
                                        inline=False)

single_checkbox = pn.widgets.Checkbox(name='Single View', value=True)

def update_single(click):
    """Toggle the rendering style radio box depending on the state of the Single View checkbox.
    
    Returns:
        None
    """
    global plotter
    single_checked = single_checkbox.value
    if single_checked is not True:
        styles_group.disabled = True
    else:
        styles_group.disabled = False
    plotter = pv.Plotter()
    click_plot(click)

# not used atm    
shadows_checkbox = pn.widgets.Checkbox(name='Shadows', value=False)

rcsb_selector_widget = pn.widgets.AutocompleteInput(name="RCSB ID", value=_rcsid, restrict=True,
                                                    placeholder="Search Here", options=RCSB_list)

# controls on sidebar
ss_props = pn.WidgetBox('# Disulfide Selection',
                        rcsb_selector_widget, rcsb_ss_widget
                        ).servable(target='sidebar')

ss_styles = pn.WidgetBox('# Rendering Styles',
                         styles_group, single_checkbox, button
                        ).servable(target='sidebar')

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")

info_md = pn.pane.Markdown("SS Info")
ss_info = pn.WidgetBox('# Disulfide Info', info_md).servable(target='sidebar')
db_info = pn.Column('### RCSB Database Info', db_md)

# Callbacks
def get_ss_idlist(event) -> list:
    """Determine the list of disulfides for the given RCSB entry and update the RCSB_ss_widget
    appropriately.
    
    Returns:
        List of SS Ids
    """
    global PDB_SS

    rcs_id = event.new
    sslist = DisulfideList([],'tmp')
    sslist = PDB_SS[rcs_id]

    idlist = [ss.name for ss in sslist]
    rcsb_ss_widget.options = idlist
    return idlist

def update_info(ss):
    src = ss.pdb_id
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info_string = f'### {name}  \n**Resolution:** {resolution:.2f} Å  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°'
    info_md.object = info_string

rcsb_selector_widget.param.watch(get_ss_idlist, 'value')
#rcsb_ss_widget.param.watch(update_info, 'value')
styles_group.param.watch(click_plot, 'value')
single_checkbox.param.watch(update_single, 'value')

def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f'## {name}'
    title_md.object = title


def update_output(ss):
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution
    
    info_string = f'**Cα-Cα:** {ss.ca_distance:.2f} Å **Cβ-Cβ:** {ss.cb_distance:.2f} Å **Torsion Length:** {ss.torsion_length:.2f}° **Resolution:** {resolution:.2f} Å **Energy:** {enrg:.2f} kcal/mol'
    output_md.object = info_string

def get_ss(event) -> Disulfide:
    global PDB_SS
    ss_id = event.new
    ss = Disulfide(PDB_SS[ss_id])
    return ss

def get_ss_id(event):
    rcsb_ss_widget.value = event.new

def render_ss(clk=True):
    global PDB_SS
    global plotter
    #global vtkpan
    global render_win

    light = True

    styles = {"Split Bonds": 'sb', "CPK":'cpk', "Ball and Stick":'bs'}

    theme = get_theme()
    if theme == 'dark':
        light = False

    ss = Disulfide()
    plotter.clear()

    ss_id = rcsb_ss_widget.value
    
    ss = PDB_SS[ss_id]
    if ss is None:
        update_output(f'Cannot find ss_id {ss_id}! Returning!')
        return

    style = styles[styles_group.value]
    single = single_checkbox.value
    #shadows = shadows_checkbox.value
    
    update_title(ss)
    update_info(ss)
    update_output(ss)

    plotter = ss.plot(plotter, single=single, style=style, shadows=False, light=light)
    vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', 
                     orientation_widget=orientation_widget,
                     enable_keybindings=enable_keybindings, min_height=500)
    
    vtkpan.param.trigger('object')

    return plotter

# Main program is trivial...

plotter = pv.Plotter()
plotter = render_ss()

vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', 
                     orientation_widget=orientation_widget,
                     enable_keybindings=enable_keybindings, min_height=500)

pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)
pn.bind(update_single, click=styles_group)

render_win = pn.Column(vtkpan).servable()
#render_win


pn.Column(
    "This example demonstrates the use of **VTK and pyvista** to display a *scene*",
    pn.Row(
        render_win
    ), min_height=500
)


