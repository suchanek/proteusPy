# RCSB Disulfide Bond Database Browser
# Author: Eric G. Suchanek, PhD.
# Last revision: 11/2/23 -egs-

import sys
import time
import proteusPy

from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList

import panel as pn
import pyvista as pv

pn.extension('vtk', sizing_mode='stretch_width', template='fast')

_vers = 0.1

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
tot = PDB_SS.TotalDisulfides
pdbs = len(PDB_SS.SSDict)
avgres = PDB_SS.Average_Resolution
 
RCSB_list = sorted(PDB_SS.IDList)
pn.state.template.param.update(title=f"RCSB Disulfide Browser {_vers}, Contains: {tot} Disulfides, {pdbs} Structures")

def get_theme() -> str:
    """Return the current theme: 'default' or 'dark'

    Returns:
        str: The current theme
    """
    args = pn.state.session_args
    if "theme" in args and args["theme"][0] == b"dark":
        return "dark"
    return "default"

def get_RCSB_info():
    """Return a string describing the curren disulfide database.

    Returns:
        str: Overall database summary string
    """
    global PDB_SS

    vers = PDB_SS.version
    tot = PDB_SS.TotalDisulfides
    pdbs = len(PDB_SS.SSDict)
    ram = (sys.getsizeof(PDB_SS.SSList) + sys.getsizeof(PDB_SS.SSDict) + sys.getsizeof(PDB_SS.TorsionDF)) / (1024 * 1024)
    res = PDB_SS.Average_Resolution
    cutoff = PDB_SS.cutoff
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(PDB_SS.timestamp))
    ssMin, ssMax = PDB_SS.SSList.minmax_energy()

    RCSB_DB_String = f'' + \
    f'    =========== RCSB Disulfide Database Summary ==============' + \
    f'       =========== Built: {timestr} ==============' + \
    f'PDB IDs present:                    {pdbs}' + \
    f'Disulfides loaded:                  {tot}' + \
    f'Average structure resolution:       {res:.2f} Å' + \
    f'Lowest Energy Disulfide:            {ssMin.name}' + \
    f'Highest Energy Disulfide:           {ssMax.name}' + \
    f'Ca distance cutoff:                 {cutoff:.2f} Å' + \
    f'Total RAM Used:                     {ram:.2f} GB.' + \
    f'    ================= proteusPy: {vers} ======================='
    return RCSB_DB_String


def click_plot(event):
    """Force a re-render of the currently selected disulfide.

    Returns:
        None
    """
    global render_win
    plotter = render_ss()
    vtkpan = pn.pane.VTK(plotter.ren_win, margin=0, sizing_mode='stretch_both', 
                         orientation_widget=orientation_widget,
                         enable_keybindings=enable_keybindings, min_height=500
                         )
    # this position is dependent on the vtk panel position in the render_win pane!
    render_win[1] = vtkpan

# Widgets

rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value=_default_ss, options=_ssidlist)

button = pn.widgets.Button(name='Render', button_type='primary')
button.on_click(click_plot)

styles_group = pn.widgets.RadioBoxGroup(name='Rending Styles', 
                                        options=['Split Bonds', 'CPK', 'Ball and Stick'], 
                                        inline=False)

single_checkbox = pn.widgets.Checkbox(name='Single View', value=True)
shadows_checkbox = pn.widgets.Checkbox(name='Shadows', value=False)
rcsb_selector_widget = pn.widgets.AutocompleteInput(name="RCSB ID", value=_rcsid, 
                                                    placeholder="Search Here", options=RCSB_list)
title_md = pn.pane.Markdown("# Title")
output_md = pn.pane.Markdown("# Output goes here")
info_md = pn.pane.Markdown("# SS Info")
db_md = pn.pane.Markdown("# Database Info goes here")

# controls on sidebar
ss_props = pn.WidgetBox('# Disulfide Selection',
                        rcsb_selector_widget, rcsb_ss_widget
                        ).servable(target='sidebar')

ss_styles = pn.WidgetBox('# Rendering Styles',
                         styles_group, single_checkbox
                        ).servable(target='sidebar')

ss_info = pn.WidgetBox('# Disulfide Statistics', info_md).servable(target='sidebar')
db_info = pn.Column('###RCSB Database Info', db_md)

# Callbacks
def get_ss_idlist(event):
    global PDB_SS

    rcs_id = event.new
    sslist = DisulfideList([],'tmp')
    sslist = PDB_SS[rcs_id]

    # print(f'RCS: {rcs_id}, |{sslist}|')
    idlist = [ss.name for ss in sslist]
    rcsb_ss_widget.options = idlist
    return idlist

rcsb_selector_widget.param.watch(get_ss_idlist, 'value')
rcsb_ss_widget.param.watch(click_plot, 'value')
styles_group.param.watch(click_plot, 'value')
single_checkbox.param.watch(click_plot, 'value')


def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f'# Disulfide: {name}'
    title_md.object = title

def update_info(ss):
    src = ss.pdb_id
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info = f'### {name}  \n**Resolution:** {resolution:.2f} Å  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°'
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

def render_ss(clk=True):
    global PDB_SS
    light = True

    styles = {"Split Bonds": 'sb', "CPK":'cpk', "Ball and Stick":'bs'}

    theme = get_theme()
    if theme == 'dark':
        light = False

    ss = Disulfide()
    plotter = pv.Plotter()
    ss_id = rcsb_ss_widget.value
    
    ss = PDB_SS[ss_id]
    if ss is None:
        update_output(f'Cannot find ss_id {ss_id}! Returning!')
        return

    style = styles[styles_group.value]
    single = single_checkbox.value
    #shadows = shadows_checkbox.value
    outputstr = f'--- Rendering: {ss_id} in style |{style}| single: {single} ---'
    
    plotter = ss.plot(single=single, style=style, shadows=False, light=light)
    
    update_title(ss)
    update_info(ss)
    update_output(outputstr)

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
