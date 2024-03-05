"""
RCSB Disulfide Bond Database Browser
Author: Eric G. Suchanek, PhD
Last revision: 1/14/2024
"""

import sys
import time
import pyvista as pv
import panel as pn

import proteusPy
from proteusPy.Disulfide import Disulfide
from proteusPy.DisulfideLoader import Load_PDB_SS
from proteusPy.DisulfideList import DisulfideList

pn.extension("vtk", sizing_mode="stretch_width", template="fast")

_vers = 0.6

# defaults for the UI

_rcsid_default = "2q7q"
_default_ss = "2q7q_75D_140D"
_ssidlist = [
    "2q7q_75D_140D",
    "2q7q_81D_113D",
    "2q7q_88D_171D",
    "2q7q_90D_138D",
    "2q7q_91D_135D",
    "2q7q_98D_129D",
    "2q7q_130D_161D",
]

_rcsid = "2q7q"
_style = "Split Bonds"
_single = True
orientation_widget = True
enable_keybindings = True

# globals
ss_state = {}
vers = tot = pdbs = 0
RCSB_list = []

# PDB_SS = Load_PDB_SS(verbose=True, subset=False)

# a few widgets
styles_group = pn.widgets.RadioBoxGroup(
    name="Rending Style", options=["Split Bonds", "CPK", "Ball and Stick"], inline=False
)

single_checkbox = pn.widgets.Checkbox(name="Single View", value=True)
rcsb_ss_widget = pn.widgets.Select(name="Disulfide", value=_default_ss, options=_ssidlist)
# not used atm
shadows_checkbox = pn.widgets.Checkbox(name="Shadows", value=False)

# rcsb_selector_widget = pn.widgets.Select(name="RCSB ID", value=_rcsid, options=RCSB_list)


rcsb_selector_widget = pn.widgets.AutocompleteInput(
    name="RCSB ID (start typing)", value=_rcsid_default, restrict=True, placeholder="Search Here", options=RCSB_list
)


button = pn.widgets.Button(name="Refresh", button_type="primary")

# controls on sidebar
ss_props = pn.WidgetBox("# Disulfide Selection", rcsb_selector_widget, rcsb_ss_widget).servable(target="sidebar")

ss_styles = pn.WidgetBox("# Rendering Styles", styles_group, single_checkbox, button).servable(target="sidebar")

# markdown panels for various text outputs
title_md = pn.pane.Markdown("Title")
output_md = pn.pane.Markdown("Output goes here")
db_md = pn.pane.Markdown("Database Info goes here")

info_md = pn.pane.Markdown("SS Info")
ss_info = pn.WidgetBox("# Disulfide Info", info_md).servable(target="sidebar")
db_info = pn.Column("### RCSB Database Info", db_md)

# pn.state.template.param.update(title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}")

# default selections
ss_state_default = {
    "single": "True",
    "style": "sb",
    "rcsb_list": "['2q7q']",
    "rcsid": "2q7q",
    "defaultss": "2q7q_75D_140D",
    "ssid_list": "['2q7q_75D_140D', '2q7q_81D_113D', '2q7q_88D_171D', '2q7q_90D_138D', \
        '2q7q_91D_135D','2q7q_98D_129D']",
}


def set_widgets_defaults():
    global RCSB_list
    styles_group.value = "Split Bonds"
    single_checkbox.value = True
    rcsb_selector_widget.options = RCSB_list
    rcsb_selector_widget.value = "2qyq"
    rcsb_ss_widget.value = "2q7q_75D_140D"
    return ss_state_default


def set_state(event):
    """
    Set the ss_state dict to the state variables and UI interaface. Push to cache.
    """
    global ss_state, _rcsid_default, _ssidlist, _default_ss, single_checkbox, styles_group

    ss_state["rcsb_list"] = RCSB_list.copy()
    ss_state["rcsid"] = _rcsid_default
    ss_state["ssid_list"] = _ssidlist.copy()
    ss_state["single"] = single_checkbox.value
    ss_state["style"] = styles_group.value
    ss_state["defaultss"] = rcsb_ss_widget.value
    print("--> Set state.")
    print_state(ss_state)
    pn.state.cache["ss_state"] = ss_state


def print_state():
    return


def load_state():
    """
    Load the state variables from the cache, update the interface.
    """
    global _ssidlist, _rcsid, _style, _single, _ssbond, _default_ss, _boot, single_checkbox, styles_group
    _ss_state = {}

    if "ss_state" in pn.state.cache:
        _ss_state = pn.state.cache["ss_state"]
        _ssidlist = _ss_state["ssid_list"]
        _rcsid = _ss_state["rcsid"]
        _style = _ss_state["style"]
        _single = _ss_state["single"]
        _default_ss = _ss_state["defaultss"]

        styles_group.value = _style
        single_checkbox.value = _single
        rcsb_selector_widget.value = _rcsid
        rcsb_ss_widget.value = _ssbond

    else:
        print(f"--> setting widgets.")
        set_widgets_defaults()
        print_state(_ss_state)

    return _ss_state


def load_data():
    global vers, tot, pdbs, RCSB_list, _boot

    _PDB_SS = Load_PDB_SS(verbose=True, subset=False)  # Load some data
    vers = _PDB_SS.version
    tot = _PDB_SS.TotalDisulfides
    pdbs = len(_PDB_SS.SSDict)
    RCSB_list = sorted(_PDB_SS.IDList)
    print(f"--> Load Data: {len(RCSB_list)}")
    # set_state(event=None)
    return _PDB_SS


if "data" in pn.state.cache:
    PDB_SS = pn.state.cache["data"]
    # vers = PDB_SS.version
    # tot = PDB_SS.TotalDisulfides
    # pdbs = len(PDB_SS.SSDict)
    # pn.state.template.param.update(title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}")
else:
    PDB_SS = pn.state.cache["data"] = load_data()
    set_widgets_defaults()
    pn.state.template.param.update(title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}")
    _boot = True

PDB_SS = pn.state.as_cached("data", load_data)
# ss_state = load_state()
# print(f'--> found state: {ss_state}')


def get_theme() -> str:
    """Returns the current theme: 'default' or 'dark'

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
    global render_win, app
    # global vtkpan

    plotter = render_ss()
    vtkpan = pn.pane.VTK(
        plotter.ren_win,
        margin=0,
        sizing_mode="stretch_both",
        orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings,
        min_height=500,
    )

    render_win[0] = vtkpan
    vtkpan.param.trigger("object")

    # this position is dependent on the vtk panel position in the render_win pane!
    print(f"RenderWin: {render_win}")


# Widgets


button.on_click(click_plot)


def update_single(click):
    """
    Toggle the rendering style radio box depending on the state of the
    Single View checkbox.

    Returns:
        None
    """
    global plotter, single_checkbox, styles_group
    single_checked = single_checkbox.value
    if single_checked is not True:
        styles_group.disabled = True
    else:
        styles_group.disabled = False
    ## plotter = pv.Plotter()
    ## click_plot(click)


# Callbacks
def get_ss_idlist(event) -> list:
    """Determine the list of disulfides for the given RCSB entry and
    update the RCSB_ss_widget appropriately.

    Returns:
        List of SS Ids
    """
    global PDB_SS

    rcs_id = event.new
    print(rcs_id)
    sslist = DisulfideList([], "tmp")
    sslist = PDB_SS[rcs_id]
    idlist = []

    if sslist:
        idlist = [ss.name for ss in sslist]
        rcsb_ss_widget.options = idlist
    print(f"--> get_ss_idlist |{rcs_id}| |{idlist}|")
    return idlist


rcsb_selector_widget.param.watch(get_ss_idlist, "value")
rcsb_ss_widget.param.watch(set_state, "value")
styles_group.param.watch(set_state, "value")
single_checkbox.param.watch(update_single, "value")


def update_title(ss):
    src = ss.pdb_id
    name = ss.name

    title = f"## {name}"
    title_md.object = title


def update_info(ss):
    src = ss.pdb_id
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info_string = f"### {name}  \n**Resolution:** {resolution:.2f} Å  \n**Energy:** {enrg:.2f} kcal/mol  \n**Cα distance:** {ss.ca_distance:.2f} Å  \n**Cβ distance:** {ss.cb_distance:.2f} Å  \n**Torsion Length:** {ss.torsion_length:.2f}°"
    info_md.object = info_string


def update_output(ss):
    enrg = ss.energy
    name = ss.name
    resolution = ss.resolution

    info_string = f"**Cα-Cα:** {ss.ca_distance:.2f} Å **Cβ-Cβ:** {ss.cb_distance:.2f} Å **Torsion Length:** {ss.torsion_length:.2f}° **Resolution:** {resolution:.2f} Å **Energy:** {enrg:.2f} kcal/mol"
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
    # global vtkpan
    global render_win

    light = True

    styles = {"Split Bonds": "sb", "CPK": "cpk", "Ball and Stick": "bs"}

    theme = get_theme()
    if theme == "dark":
        print("--> Dark")
        light = False

    ss = Disulfide()
    plotter.clear()

    ss_id = rcsb_ss_widget.value

    ss = PDB_SS[ss_id]
    if ss is None:
        update_output(f"Cannot find ss_id {ss_id}! Returning!")
        return

    style = styles[styles_group.value]
    single = single_checkbox.value
    # shadows = shadows_checkbox.value

    update_title(ss)
    update_info(ss)
    update_output(ss)

    plotter = ss.plot(plotter, single=single, style=style, shadows=False, light=light)
    vtkpan = pn.pane.VTK(
        plotter.ren_win,
        margin=0,
        sizing_mode="stretch_both",
        orientation_widget=orientation_widget,
        enable_keybindings=enable_keybindings,
        min_height=500,
    )

    vtkpan.param.trigger("object")

    return plotter


def on_theme_change(event):
    selected_theme = event.obj.theme
    print(f"--> Theme Change: {selected_theme}")


plotter = pv.Plotter()
plotter = render_ss()

vtkpan = pn.pane.VTK(
    plotter.ren_win,
    margin=0,
    sizing_mode="stretch_both",
    orientation_widget=orientation_widget,
    enable_keybindings=enable_keybindings,
    min_height=500,
)

pn.bind(get_ss_idlist, rcs_id=rcsb_selector_widget)
pn.bind(update_single, click=styles_group)

render_win = pn.Column(vtkpan)
render_win.servable()


"""

# Instantiate the template with widgets displayed in the sidebar
app = pn.template.FastListTemplate(
    title=f"RCSB Disulfide Browser: {tot:,} Disulfides, {pdbs:,} Structures, V{vers}",
    #sidebar=[ss_styles, ss_props],
)
# Append a layout to the main area, to demonstrate the list-like API
app.main.append(
    pn.Row(
        render_win,
    )
)
app.servable();


app = pn.panel(
    pn.Column(
        "This example demonstrates the use of **VTK and pyvista** to display a *scene*",
        pn.Row(
            render_win
        ), min_height=500
    )
)

app.servable()
"""
